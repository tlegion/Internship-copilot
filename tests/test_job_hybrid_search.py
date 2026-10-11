import hashlib
import json
import math
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import APIConnectionError

from errors import ProviderError, StorageError, ToolError
from rag_lab import job_chunking, job_hybrid_search, job_indexing
from rag_lab import reranker
from scripts import migrate_db
from storages import database, job_index
from tools.jobs import search_jobs_hybrid
from test_list_jobs import tool_call

ROOT = Path(__file__).resolve().parents[1]


def fake_embeddings(texts):
    return [[1.0, 0.0] if "Python" in text else [0.0, 1.0] for text in texts]


class JobChunkTests(unittest.TestCase):
    def job(self, description):
        return dict(job_id="job_009", title="Backend Intern", company="Demo",
                    location="Sydney", description=description)

    def test_short_chunk_and_deterministic_hash(self):
        job = self.job("Build Python APIs.")
        chunks = job_chunking.chunk_job(job)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks, job_chunking.chunk_job(job))
        chunk = chunks[0]
        self.assertNotIn("job_009", chunk["text"])
        for value in ("Backend Intern", "Demo", "Sydney", job["description"]):
            self.assertIn(value, chunk["text"])
        self.assertEqual(chunk["content_hash"], hashlib.sha256(chunk["text"].encode()).hexdigest())
        self.assertEqual(chunk["chunk_id"], "job_009_chunk_000")

    def test_long_job_bounded_chunks_context_and_overlap(self):
        job = self.job("\n\n".join(f"Sentence {i}: build Python APIs and RAG tools." for i in range(500)))
        chunks = job_chunking.chunk_job(job)
        self.assertGreater(len(chunks), 2)
        prefix = job_chunking.job_header(job)
        rebuilt = chunks[0]["text"][len(prefix):]
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["job_id"], job["job_id"])
            self.assertEqual(chunk["chunk_index"], index)
            self.assertTrue(chunk["text"].startswith(prefix))
            self.assertLessEqual(job_chunking.count_tokens(chunk["text"]), 800)
            if index:
                previous = chunks[index - 1]["text"][len(prefix):]
                current = chunk["text"][len(prefix):]
                overlap = max(length for length in range(1, min(len(previous), len(current)) + 1)
                              if previous[-length:] == current[:length])
                self.assertGreaterEqual(job_chunking.count_tokens(current[:overlap]), 45)
                self.assertLessEqual(job_chunking.count_tokens(current[:overlap]), 50)
                rebuilt += current[overlap:]
        self.assertEqual(rebuilt, job["description"])

    def test_unicode_special_tokens_and_large_header(self):
        job = self.job(("你好🙂 <|endoftext|> Python " * 1000))
        job["title"] = "Long title " * 1000
        chunks = job_chunking.chunk_job(job)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertNotIn("\ufffd", chunk["text"])
            self.assertLessEqual(job_chunking.count_tokens(chunk["text"]), 800)

    def test_null_and_empty_location_have_different_hashes(self):
        job = self.job("Python")
        job["location"] = None
        first = job_chunking.chunk_job(job)
        job["location"] = ""
        self.assertNotEqual(first, job_chunking.chunk_job(job))

    def test_unusual_token_boundary_declines_oversized_piece_without_looping(self):
        job = self.job("Python APIs. " * 1000)
        header = job_chunking.job_header(job)
        real_count = job_chunking.count_tokens

        def changed_boundary_count(text):
            if text == header + "decoded piece":
                return 801
            return real_count(text)

        with patch.object(job_chunking, "split_long_paragraph", return_value=["decoded piece"]), \
                patch.object(job_chunking, "count_tokens", side_effect=changed_boundary_count):
            with self.assertRaisesRegex(ValueError, "Decoded chunk exceeds"):
                job_chunking.chunk_job(job)
        # Normal Unicode token boundaries still produce nonempty, bounded chunks.
        for chunk in job_chunking.chunk_job(self.job("你好🙂 " * 1000)):
            self.assertTrue(chunk["text"].strip())
            self.assertLessEqual(real_count(chunk["text"]), 800)


class JobIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "jobs.db"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            for migration in migrate_db.MIGRATIONS:
                connection.executescript(migration.read_text())
            connection.execute("INSERT INTO evidence VALUES ('e1', 'c1', 's1', 'project', NULL, 'Evidence Python')")
            connection.execute("INSERT INTO chunks VALUES ('e1_chunk_000', 'e1', 0, 'Evidence Python', 'ehash')")
            connection.execute("INSERT INTO chunk_embeddings VALUES ('ehash', 'model', 2, '[1,0]')")
            connection.execute("INSERT INTO chunk_fts VALUES ('e1_chunk_000', 'Evidence Python')")
            connection.executemany("INSERT INTO jobs VALUES (?, ?, ?, ?, ?)", [
                ("job_001", "Python Intern", "Demo", "Build Python APIs and RAG.\nFull original details.", "Sydney"),
                ("job_002", "Backend Intern", "Other", "Work with Rust and Kubernetes.", "Sydney"),
                ("job_003", "Python Intern", "Demo", "Build Python APIs and RAG.\nFull original details.", "Sydney"),
                ("job_004", "Python Intern", "Demo", "Python services for London.", "London"),
                ("job_005", "Intern", "Other", "Java service development.", None),
            ])
            connection.commit()
        self.start_patch(database, "DB_PATH", self.path)
        self.embed_index = self.start_patch(job_indexing, "embedding_texts", side_effect=fake_embeddings)
        self.embed_query = self.start_patch(job_hybrid_search, "embedding_texts", side_effect=fake_embeddings)
        self.start_patch(job_hybrid_search, "rerank_results",
                         side_effect=lambda query, candidates, top_k, **kwargs: candidates[:top_k])

    def start_patch(self, target, name, *args, **kwargs):
        patcher = patch.object(target, name, *args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def execute(self, sql, parameters=()):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            return connection.execute(sql, parameters).fetchall()

    def index_snapshot(self):
        return [self.execute(f"SELECT * FROM {table} ORDER BY 1") for table in
                ("chunks_jobs", "chunk_embeddings_jobs", "chunk_fts_jobs")]

    def sync(self):
        return job_indexing.sync_job_index()

    def test_v003_foreign_keys_unique_constraint_and_migration_registration(self):
        self.assertEqual(migrate_db.MIGRATIONS[-1].name, "V003__add_rag_job.sql")
        self.assertEqual(self.execute("PRAGMA foreign_key_check"), [])
        with self.assertRaises(sqlite3.IntegrityError):
            self.execute("INSERT INTO chunks_jobs VALUES ('bad', 'missing', 0, 'text', 'hash')")
        self.sync()
        with self.assertRaises(sqlite3.IntegrityError):
            self.execute("INSERT INTO chunks_jobs VALUES ('duplicate', 'job_001', 0, 'text', 'hash')")

    def test_identical_text_cache_reuse_and_repeated_sync(self):
        first = self.sync()
        self.assertEqual(first, dict(jobs=5, chunks=5, cached_embeddings=0, new_embeddings=4))
        self.assertEqual(len(self.embed_index.call_args.args[0]), 4)
        snapshot = self.index_snapshot()
        self.embed_index.reset_mock()
        self.assertEqual(self.sync(), dict(jobs=5, chunks=5, cached_embeddings=4, new_embeddings=0))
        self.embed_index.assert_not_called()
        self.assertEqual(self.index_snapshot(), snapshot)
        self.assertEqual(self.execute("SELECT chunk_id, text FROM chunks_jobs ORDER BY chunk_id"),
                         self.execute("SELECT chunk_id, text FROM chunk_fts_jobs ORDER BY chunk_id"))

    def test_cache_lookup_stays_below_sqlite_parameter_limit(self):
        hashes = [f"hash_{index}" for index in range(1100)]
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executemany("INSERT INTO chunk_embeddings_jobs VALUES (?, ?, 2, '[1,0]')",
                                   [(value, job_chunking.EMBEDDING_MODEL) for value in hashes])

        def limited_connection():
            connection = database.db_connection_ro()
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
            return connection

        with patch.object(job_index, "db_connection_ro", side_effect=limited_connection):
            self.assertEqual(job_index.get_cached_job_hashes(hashes, job_chunking.EMBEDDING_MODEL), set(hashes))

    def test_evidence_tables_and_search_unchanged(self):
        tables = ("evidence", "chunks", "chunk_embeddings", "chunk_fts")
        before = [self.execute(f"SELECT * FROM {table}") for table in tables]
        self.sync()
        search_jobs_hybrid("Python", ["Python"])
        self.assertEqual(before, [self.execute(f"SELECT * FROM {table}") for table in tables])
        from rag_lab.bm25_search import bm25_search
        self.assertEqual(bm25_search(["Python"])[0]["evidence_id"], "e1")

    def test_write_failure_rolls_back_full_index(self):
        self.sync()
        snapshot = self.index_snapshot()
        self.execute("UPDATE jobs SET description = 'New skills' WHERE id = 'job_001'")
        self.execute("CREATE TRIGGER fail_embedding BEFORE INSERT ON chunk_embeddings_jobs BEGIN SELECT RAISE(ABORT, 'simulated'); END")
        with self.assertRaises(StorageError):
            self.sync()
        self.assertEqual(self.index_snapshot(), snapshot)

    def test_embedding_failure_does_not_change_index(self):
        self.sync()
        snapshot = self.index_snapshot()
        self.execute("UPDATE jobs SET description = 'New text' WHERE id = 'job_001'")
        self.embed_index.side_effect = ProviderError("simulated")
        with self.assertRaises(ProviderError):
            self.sync()
        self.assertEqual(self.index_snapshot(), snapshot)

    def test_embedding_count_mismatch_does_not_commit(self):
        self.embed_index.side_effect = lambda texts: []
        with self.assertRaises(ProviderError):
            self.sync()
        self.assertEqual(self.index_snapshot(), [[], [], []])

    def test_invalid_provider_embeddings_do_not_commit(self):
        for vector in ([], [0, 0], [float("nan"), 1], ["invalid", 1]):
            with self.subTest(vector=vector):
                self.embed_index.side_effect = lambda texts: [vector for _ in texts]
                with self.assertRaises(ProviderError):
                    self.sync()
                self.assertEqual(self.index_snapshot(), [[], [], []])

    def test_invalid_cached_embedding_follows_storage_error_contract(self):
        self.sync()
        self.execute("UPDATE chunk_embeddings_jobs SET embedding = '[0,0]'")
        with self.assertRaises(StorageError):
            search_jobs_hybrid("Python", [])

    def test_invalid_query_embedding_follows_provider_error_contract(self):
        self.sync()
        self.embed_query.side_effect = lambda texts: [[float("nan"), 1]]
        with self.assertRaises(ProviderError):
            search_jobs_hybrid("Python", [])

    def test_migration_runner_applies_v003_once(self):
        migration_path = self.path.parent / "migration.db"
        with closing(sqlite3.connect(migration_path)) as connection:
            for migration in migrate_db.MIGRATIONS[:-1]:
                connection.executescript(migration.read_text())
            connection.execute("PRAGMA user_version = 3")
        with patch.object(migrate_db, "DB_PATH", migration_path), redirect_stdout(io.StringIO()):
            migrate_db.migrate_db()
            migrate_db.migrate_db()
        with closing(sqlite3.connect(migration_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM chunks_jobs").fetchone()[0], 0)

    def test_real_evidence_tool_keeps_hybrid_and_reranker_arguments(self):
        from tools import evidence
        with patch.object(evidence, "hybrid_search", return_value=[{"text": "hit"}]) as hybrid, \
                patch.object(evidence, "rerank_results", return_value=[{"text": "ranked"}]) as rerank, \
                redirect_stdout(io.StringIO()):
            result = evidence.search_evidence_hybrid("Python", ["Python"], "c1")
        hybrid.assert_called_once_with(query="Python", keywords=["Python"], candidate_id="c1", top_k=10)
        rerank.assert_called_once_with(query="Python", candidates=[{"text": "hit"}], top_k=3)
        self.assertEqual(result, [{"text": "ranked"}])

    def test_job_change_during_embedding_aborts_save(self):
        def change_jobs(texts):
            self.execute("UPDATE jobs SET company = 'Changed' WHERE id = 'job_001'")
            return fake_embeddings(texts)
        self.embed_index.side_effect = change_jobs
        with self.assertRaisesRegex(ToolError, "Jobs changed during indexing"):
            self.sync()
        self.assertEqual(self.index_snapshot(), [[], [], []])

    def test_new_changed_and_deleted_jobs_require_sync(self):
        self.sync()
        for sql in (
            "INSERT INTO jobs VALUES ('job_006', 'Intern', 'Demo', 'Python RAG', 'Sydney')",
            "UPDATE jobs SET description = 'Updated Python' WHERE id = 'job_001'",
            "DELETE FROM jobs WHERE id = 'job_002'",
        ):
            self.execute(sql)
            with self.assertRaisesRegex(ToolError, "sync_job_index.py"):
                search_jobs_hybrid("Python", ["Python"])
            self.embed_query.assert_not_called()
            self.sync()
        result = search_jobs_hybrid("Python", ["Python"])
        self.assertIn("job_006", [job["job_id"] for job in result["jobs"]])
        self.assertNotIn("job_002", [job["job_id"] for job in result["jobs"]])
        self.assertEqual(self.execute("SELECT COUNT(*) FROM chunk_fts_jobs WHERE chunk_id LIKE 'job_002_%'"), [(0,)])

    def test_incomplete_or_corrupt_fts_and_missing_embeddings_are_stale(self):
        for sql in ("DELETE FROM chunk_fts_jobs", "UPDATE chunk_fts_jobs SET text = 'wrong'",
                    "DELETE FROM chunk_embeddings_jobs"):
            with self.subTest(sql=sql):
                self.sync()
                self.execute(sql)
                with self.assertRaises(ToolError):
                    search_jobs_hybrid("Python", [])

    def test_filters_restrict_both_branches_and_complete_results(self):
        self.sync()
        for filters, expected in (({"company": "Demo"}, {"job_001", "job_003", "job_004"}),
                                  ({"location": "Sydney"}, {"job_001", "job_002", "job_003"}),
                                  ({"company": "Demo", "location": "Sydney"}, {"job_001", "job_003"})):
            with self.subTest(filters=filters):
                dense, bm25 = job_index.load_job_search_data(job_chunking.EMBEDDING_MODEL,
                                                            '"Python"', filters.get("company"), filters.get("location"))
                self.assertEqual({row["job_id"] for row in dense}, expected)
                self.assertLessEqual({row["job_id"] for row in bm25}, expected)
                result = search_jobs_hybrid("Python APIs", ["Python"], **filters)
                self.assertEqual({job["job_id"] for job in result["jobs"]}, expected)
                self.assertEqual([job["rank"] for job in result["jobs"]], list(range(1, len(expected) + 1)))
                for job in result["jobs"]:
                    self.assertEqual(set(job), {"job_id", "title", "company", "description", "location", "rank"})
                    self.assertEqual(job["description"], database.get_job(job["job_id"])["description"])
                self.assertEqual(json.loads(json.dumps(result)), result)

    def test_empty_keywords_dense_only_and_one_query_embedding(self):
        self.sync()
        result = search_jobs_hybrid("Python APIs", [], top_k=2)
        self.assertEqual(len(result["jobs"]), 2)
        self.embed_query.assert_called_once_with(["Python APIs"])
        self.assertTrue(all("Python" in job["description"] for job in result["jobs"]))

    def test_bm25_technical_keyword_and_safe_literals(self):
        self.sync()
        from rag_lab.bm25_search import build_fts_query
        _, hits = job_index.load_job_search_data(job_chunking.EMBEDDING_MODEL,
                                                build_fts_query(["Kubernetes"]), None, None)
        self.assertEqual([hit["job_id"] for hit in hits], ["job_002"])
        for keywords in ([], [""], ['" OR Python'], ["Python) OR (Rust"]):
            search_jobs_hybrid("Python", keywords)

    def test_many_chunks_deduplicate_without_underfilling_jobs(self):
        self.execute("UPDATE jobs SET description = ? WHERE id = 'job_001'",
                     ("Python RAG services and APIs. " * 3000,))
        self.sync()
        self.assertGreater(self.execute("SELECT COUNT(*) FROM chunks_jobs WHERE job_id = 'job_001'")[0][0], 10)
        result = search_jobs_hybrid("Python", [], top_k=5)
        self.assertEqual(len(result["jobs"]), 5)
        self.assertEqual(len({job["job_id"] for job in result["jobs"]}), 5)

    def test_empty_eligible_set_and_empty_database(self):
        self.sync()
        self.assertEqual(search_jobs_hybrid("Python", [], company="Demo' OR 1=1 --"), {"jobs": []})
        self.embed_query.assert_not_called()
        self.execute("DELETE FROM jobs")
        self.sync()
        self.assertEqual(search_jobs_hybrid("Python", []), {"jobs": []})

    def test_stale_check_is_global_even_with_empty_filter(self):
        self.sync()
        self.execute("UPDATE jobs SET title = 'Changed' WHERE id = 'job_001'")
        with self.assertRaises(ToolError):
            search_jobs_hybrid("Python", [], company="Missing")

    def test_registration_strict_schema_dispatch_and_null_default(self):
        self.sync()
        schema = next(tool for tool in tool_call.tools if tool["name"] == "search_jobs_hybrid")
        self.assertTrue(schema["strict"])
        params = schema["parameters"]
        self.assertFalse(params["additionalProperties"])
        self.assertEqual(set(params["required"]), set(params["properties"]))
        for field in ("company", "location", "top_k"):
            self.assertIn("null", params["properties"][field]["type"])
        with patch.object(tool_call, "request_tool_approval") as approve:
            result = tool_call.run_tool("search_jobs_hybrid", dict(query="Python", keywords=[], company=None, location=None, top_k=None), "call_job")
        approve.assert_not_called()
        self.assertTrue(result["success"])
        self.assertEqual(len(result["result"]["jobs"]), 5)
        self.assertNotIn("sync_job_index", tool_call.tool_list)

    def test_invalid_inputs_follow_tool_error_contract(self):
        for arguments in ({"query": ""}, {"query": "  "}, {"query": None},
                          {"keywords": None}, {"keywords": "Python"}, {"keywords": [1]},
                          {"company": 1}, {"location": []}, {"top_k": 0}, {"top_k": -1},
                          {"top_k": 21}, {"top_k": True}, {"top_k": 1.5}, {"top_k": "10"}):
            with self.subTest(arguments=arguments):
                result = tool_call.run_tool("search_jobs_hybrid", {"query": "Python", "keywords": [], **arguments}, "invalid")
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], "ToolError")
        self.embed_query.assert_not_called()


class FusionTests(unittest.TestCase):
    def test_weighted_rrf_and_debug_scores(self):
        dense = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.5}]
        bm25 = [{"chunk_id": "b", "score": -2}, {"chunk_id": "a", "score": -1}]
        result = job_hybrid_search.fuse_job_chunks(dense, bm25)
        self.assertEqual([row["chunk_id"] for row in result], ["a", "b"])
        self.assertTrue(math.isclose(result[0]["rrf_score"], 0.9))
        self.assertTrue(math.isclose(result[1]["rrf_score"], 0.6))
        self.assertEqual(result[1]["bm25_score"], -2)


class JobRerankTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for index in range(30):
            self.rows.append({
                "chunk_id": f"job_{index:03d}_chunk_000", "job_id": f"job_{index:03d}",
                "title": f"Python Intern {index}", "company": "Demo", "location": "Sydney",
                "description": f"Complete original description {index}. " * 100,
                "text": f"Matching Python RAG passage {index}", "score": 1.0,
            })
        # A second chunk from the best job must never become a second candidate.
        self.rows.insert(1, {**self.rows[0], "chunk_id": "job_000_chunk_001", "text": "Lower ranked passage"})
        self.start_patch(job_hybrid_search, "load_job_search_data", return_value=(self.rows, []))
        self.start_patch(job_hybrid_search, "dense_job_search", side_effect=lambda query, chunks: chunks)
        self.client = self.start_patch(reranker, "OpenAI").return_value
        environment = patch.dict(reranker.os.environ, {"OPENAI_API_KEY": "unit-test-key"})
        environment.start()
        self.addCleanup(environment.stop)

    def start_patch(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def search(self, top_k=2):
        return job_hybrid_search.hybrid_job_search("Python RAG backend roles", ["Python"], None, None, top_k)

    def parsed(self, indices):
        self.client.responses.parse.return_value = SimpleNamespace(
            output_parsed=SimpleNamespace(ranked_indices=indices)
        )

    def test_unique_bounded_candidates_and_matching_metadata(self):
        with patch.object(job_hybrid_search, "rerank_results",
                          side_effect=lambda query, candidates, top_k, **kwargs: candidates[:top_k]) as rerank:
            self.search()
        query, candidates = rerank.call_args.args
        self.assertEqual(query, "Python RAG backend roles")
        self.assertEqual(len(candidates), 25)
        self.assertEqual(len({candidate["job_id"] for candidate in candidates}), 25)
        self.assertEqual(rerank.call_args.kwargs, {"top_k": 2, "job_mode": True})
        text = candidates[0]["text"]
        for value in ("Python Intern 0", "Demo", "Sydney", "Matching Python RAG passage 0"):
            self.assertIn(value, text)
        self.assertNotIn("Lower ranked passage", text)
        self.assertNotIn("Complete original description", text)

    def test_llm_changes_order_preserves_full_jobs_and_reassigns_ranks(self):
        self.parsed([24, 0])
        result = self.search()
        self.assertEqual([job["job_id"] for job in result["jobs"]], ["job_024", "job_000"])
        self.assertEqual([job["rank"] for job in result["jobs"]], [1, 2])
        self.assertEqual(result["jobs"][0]["description"], self.rows[25]["description"])
        self.assertEqual(set(result["jobs"][0]), {"job_id", "title", "company", "location", "description", "rank"})
        self.assertEqual(json.loads(json.dumps(result)), result)
        request = self.client.responses.parse.call_args.kwargs
        self.assertEqual(request["model"], reranker.RERANK_MODEL)
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertIn("not candidate-job fit", request["input"][1]["content"])

    def test_top_k_contract(self):
        self.parsed(list(range(19, -1, -1)))
        self.assertEqual(len(self.search(top_k=20)["jobs"]), 20)
        self.parsed([5])
        self.assertEqual(len(self.search(top_k=1)["jobs"]), 1)

    def test_invalid_indices_fall_back_to_distinct_rrf_order(self):
        for indices in ([0, 0], [-1, 0], [0, 25], [0], []):
            with self.subTest(indices=indices):
                self.parsed(indices)
                output = io.StringIO()
                with redirect_stdout(output):
                    result = self.search()
                self.assertEqual([job["job_id"] for job in result["jobs"]], ["job_000", "job_001"])
                self.assertEqual([job["rank"] for job in result["jobs"]], [1, 2])
                self.assertIn("reranking was not applied", output.getvalue())

    def test_api_failure_missing_parse_and_invalid_parse_fall_back(self):
        failures = (APIConnectionError(request=Mock()),
                    ValueError("invalid structured output"))
        for failure in failures:
            with self.subTest(failure=failure), redirect_stdout(io.StringIO()):
                self.client.responses.parse.side_effect = failure
                self.assertEqual([job["job_id"] for job in self.search()["jobs"]], ["job_000", "job_001"])
        self.client.responses.parse.side_effect = None
        self.client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
        with redirect_stdout(io.StringIO()):
            self.assertEqual([job["job_id"] for job in self.search()["jobs"]], ["job_000", "job_001"])

    def test_unavailable_configuration_falls_back_without_api_call(self):
        with patch.dict(reranker.os.environ, {"OPENAI_API_KEY": ""}), redirect_stdout(io.StringIO()):
            result = self.search()
        self.assertEqual([job["job_id"] for job in result["jobs"]], ["job_000", "job_001"])
        self.client.responses.parse.assert_not_called()

    def test_empty_results_do_not_rerank(self):
        with patch.object(job_hybrid_search, "load_job_search_data", return_value=([], [])), \
                patch.object(job_hybrid_search, "rerank_results") as rerank:
            self.assertEqual(self.search(), {"jobs": []})
        rerank.assert_not_called()

    def test_evidence_defaults_and_fallback_remain_unchanged(self):
        candidates = [{"text": "Evidence A"}, {"text": "Evidence B"}]
        self.parsed([1])
        result = reranker.rerank_results("Python experience", candidates, top_k=1)
        self.assertEqual(result, [{"text": "Evidence B", "rerank_rank": 1}])
        request = self.client.responses.parse.call_args.kwargs
        self.assertIn("Candidate evidence:", request["input"][1]["content"])
        self.assertIn("Rank candidate evidence", request["input"][0]["content"])
        self.client.responses.parse.side_effect = APIConnectionError(request=Mock())
        self.assertEqual(reranker.rerank_results("Python experience", candidates, top_k=1), candidates[:1])


if __name__ == "__main__":
    unittest.main()
