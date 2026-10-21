from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openai import APIConnectionError
from pydantic import ValidationError

from errors import ProviderError, StorageError, ToolError
import loop
from models import ModelProvider, ModelToolCall
from rag_lab import dense_search, hybrid_search, reranker
from rag_lab.chunking import chunk_documents
from scripts.migrate_db import MIGRATIONS
from skill import discover_skills
from storages import database
import tool_call
from tools import candidate


ROOT = Path(__file__).resolve().parents[1]


def response(parsed):
    return SimpleNamespace(status="completed", output_parsed=parsed)


class CandidateRecognitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "candidate.db"
        self.documents = [
            dict(evidence_id="e1", candidate_id="c1", source_id="university-project",
                 category="project", date="2025-06",
                 text="Built a REST API using Django for a university project. This was not a production deployment."),
            dict(evidence_id="e2", candidate_id="c1", source_id="old-record",
                 category="resume", date="2024-01", text="I have no SQL experience as of January 2024."),
            dict(evidence_id="e3", candidate_id="c1", source_id="recent-project",
                 category="project", date="2026-06", text="Completed a PostgreSQL project in June 2026."),
            dict(evidence_id="e4", candidate_id="c1", source_id="deployment-correction",
                 category="correction", date="2026-07",
                 text="The claimed production deployment was actually a simulated lab exercise."),
            dict(evidence_id="other", candidate_id="c2", source_id="foreign",
                 category="project", date="2026-08", text="Built a production Django API for another candidate."),
        ]
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            for migration in MIGRATIONS:
                connection.executescript(migration.read_text(encoding="utf-8"))
            connection.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                                   [tuple(document.values()) for document in self.documents])
            for chunk in chunk_documents(self.documents):
                digest = hashlib.sha256(chunk["text"].encode()).hexdigest()
                connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                                   (chunk["chunk_id"], chunk["evidence_id"], chunk["chunk_index"], chunk["text"], digest))
                connection.execute("INSERT OR IGNORE INTO chunk_embeddings VALUES (?, ?, ?, ?)",
                                   (digest, candidate.EMBEDDING_MODEL, 2, "[1.0, 0.0]"))
                connection.execute("INSERT INTO chunk_fts VALUES (?, ?)", (chunk["chunk_id"], chunk["text"]))
        self.start_patch(patch.object(database, "DB_PATH", self.db_path))
        self.start_patch(patch.object(candidate, "load_dotenv"))
        self.start_patch(patch.dict(candidate.os.environ, {"OPENAI_API_KEY": "test-only"}))
        self.client = self.start_patch(patch.object(candidate, "OpenAI")).return_value
        self.embed = self.start_patch(patch.object(dense_search, "embedding_texts", return_value=[[1.0, 0.0]]))
        self.rerank = self.start_patch(patch.object(reranker, "rerank_results",
            side_effect=AssertionError("Candidate retrieval must not rerank")))
        self.fusion = self.start_patch(patch.object(hybrid_search, "reciprocal_rank_fusion",
            side_effect=AssertionError("Candidate retrieval must not use RRF")))
        self.catalog = discover_skills()
        self.loaded = {"candidate-recognition"}

    def start_patch(self, patcher):
        self.addCleanup(patcher.stop)
        return patcher.start()

    def assessment(self, verdict="supported", citations=None, **kwargs):
        if citations is None:
            citations = [dict(evidence_id="e1", role="supports",
                              quote="Built a REST API using Django for a university project.")]
        return candidate.ClaimAssessment(
            is_atomic=kwargs.get("is_atomic", True), verdict=verdict,
            reason=kwargs.get("reason", "The original source directly establishes the scoped claim."),
            selected_evidence=[candidate.SelectedEvidence(**item) for item in citations],
        )

    def batch(self, claims=None):
        if claims is None:
            claims = [("The candidate built a Django API for a university project.", "e1")]
        return candidate.ClaimBatch(claims=[candidate.AtomicClaim(claim=text, source_evidence_id=source)
                                            for text, source in claims])

    def call(self, name="verify_candidate_claim", arguments=None, loaded=None):
        if arguments is None:
            arguments = {"candidate_id": "c1", "claim": "I built a Django API for a university project."}
        return tool_call.run_tool(name, arguments, "call", catalog=self.catalog,
                                  loaded_skills=self.loaded if loaded is None else loaded)

    def chunks(self, *ids):
        documents = {d["evidence_id"]: d for d in self.documents}
        return [dict(chunk_id=f"{key}_chunk_{index}", evidence_id=key,
                     candidate_id=documents[key]["candidate_id"], text=documents[key]["text"])
                for index, key in enumerate(ids)]

    def test_tools_are_gated_read_only_and_internal_helpers_are_not_registered(self):
        names = {"build_candidate_profile", "verify_candidate_claim"}
        base = set(tool_call.BASE_TOOL_LIST)
        self.assertIn("search_evidence_hybrid", base)
        self.assertTrue(names.isdisjoint(base))
        for loaded in (set(), {"job-comparison"}, {"missing"}):
            available = tool_call.get_available_tools(self.catalog, loaded)
            self.assertTrue(names.isdisjoint(available))
            for name in names:
                self.assertFalse(self.call(name, {"candidate_id": "c1"}, loaded)["success"])
        available = tool_call.get_available_tools(self.catalog, self.loaded)
        self.assertTrue(names.issubset(available))
        for name in names:
            self.assertFalse(available[name]["requires_approval"])
        self.assertTrue(base.issubset(available))
        for name in ("_retrieve_evidence_ids", "_check_claim", "_extract_claims"):
            self.assertNotIn(name, available)
            self.assertFalse(self.call(name, {})["success"])
        self.client.responses.parse.assert_not_called()

    def test_loading_only_activates_tools_in_next_batch(self):
        self.client.responses.parse.return_value = response(self.assessment())
        loaded = set()
        calls = [ModelToolCall("load_skill", {"name": "candidate-recognition"}, "load"),
                 ModelToolCall("verify_candidate_claim", {"candidate_id": "c1", "claim": "I built a Django API."}, "verify"),
                 ModelToolCall("build_candidate_profile", {"candidate_id": "c1"}, "profile")]
        first = loop.tool_loop(calls, ModelProvider.OPENAI, 1, 1, catalog=self.catalog, loaded_skills=loaded)
        self.assertEqual([item.output["success"] for item in first], [True, False, False])
        self.assertEqual(loaded, self.loaded)
        self.client.responses.parse.assert_not_called()
        with patch.object(tool_call, "request_tool_approval") as approve:
            second = loop.tool_loop(calls[1:2], ModelProvider.OPENAI, 1, 2,
                                    catalog=self.catalog, loaded_skills=loaded)
        self.assertTrue(second[0].output["success"])
        approve.assert_not_called()

    def test_failed_loading_does_not_activate_tools(self):
        loaded = set()
        with patch.object(tool_call, "load_skill", side_effect=ToolError("Unreadable Skill")):
            outputs = loop.tool_loop([ModelToolCall("load_skill", {"name": "candidate-recognition"}, "load")],
                                     ModelProvider.OPENAI, 1, 1, catalog=self.catalog, loaded_skills=loaded)
        self.assertFalse(outputs[0].output["success"])
        self.assertEqual(loaded, set())

    def test_generated_schemas_match_public_parameters(self):
        schemas = {item["name"]: item for item in tool_call.get_tool_schemas(self.catalog, self.loaded)}
        for name, fields in (("build_candidate_profile", {"candidate_id"}),
                             ("verify_candidate_claim", {"candidate_id", "claim"})):
            schema = schemas[name]
            self.assertTrue(schema["strict"])
            self.assertEqual(set(schema["parameters"]["properties"]), fields)
            self.assertEqual(set(schema["parameters"]["required"]), fields)
            self.assertFalse(schema["parameters"]["additionalProperties"])
            for field in schema["parameters"]["properties"].values():
                self.assertEqual(field["type"], "string")
                self.assertTrue(field["description"])

    def test_invalid_tool_arguments_never_reach_storage_or_approval(self):
        with patch.object(candidate, "get_evidence_documents") as storage, \
                patch.object(tool_call, "request_tool_approval") as approve:
            for name, arguments in (
                ("build_candidate_profile", {}), ("build_candidate_profile", {"candidate_id": None}),
                ("build_candidate_profile", {"candidate_id": " "}),
                ("verify_candidate_claim", {"candidate_id": "c1"}),
                ("verify_candidate_claim", {"candidate_id": "c1", "claim": []}),
                ("verify_candidate_claim", {"candidate_id": "c1", "claim": ""}),
                ("verify_candidate_claim", {"candidate_id": "c1", "claim": "  "}),
                ("verify_candidate_claim", {"candidate_id": "c1", "claim": "API", "catalog": {}}),
            ):
                with self.subTest(name=name, arguments=arguments):
                    self.assertEqual(self.call(name, arguments)["error_type"], "ToolError")
            storage.assert_not_called()
            approve.assert_not_called()

    def test_unknown_candidate_without_evidence_is_explicit_error(self):
        for name in ("build_candidate_profile", "verify_candidate_claim"):
            result = self.call(name, {"candidate_id": "unknown", **({"claim": "SQL"} if name.startswith("verify") else {})})
            self.assertEqual(result["error_type"], "ToolError")
            self.assertIn("No evidence", result["error_msg"])
        self.client.responses.parse.assert_not_called()

    def test_profile_extracts_complete_originals_and_uses_two_loops(self):
        batch = self.batch([("I built a Django API for a university project.", "e1"),
                            ("I completed a PostgreSQL project in 2026.", "e3")])
        self.client.responses.parse.side_effect = [response(batch), response(self.assessment()),
            response(self.assessment(citations=[dict(evidence_id="e3", role="supports",
                                                    quote=self.documents[2]["text"])]))]
        events = []
        retrieve = candidate._retrieve_evidence_ids
        check = candidate._check_claim

        def tracked_retrieve(*args):
            events.append("retrieve")
            return retrieve(*args)

        def tracked_check(*args):
            events.append("check")
            return check(*args)

        with patch.object(candidate, "_retrieve_evidence_ids", side_effect=tracked_retrieve), \
                patch.object(candidate, "_check_claim", side_effect=tracked_check), \
                patch.object(tool_call, "request_tool_approval") as approve:
            result = self.call("build_candidate_profile", {"candidate_id": "c1"})
        self.assertTrue(result["success"])
        self.assertEqual(events, ["retrieve", "retrieve", "check", "check"])
        self.assertEqual(len(result["result"]["assessments"]), 2)
        self.assertEqual(result["result"]["assessments"][0]["source_evidence_id"], "e1")
        self.assertEqual(self.client.responses.parse.call_count, 3)
        payload = json.loads(self.client.responses.parse.call_args_list[0].kwargs["input"][1]["content"])
        self.assertEqual(payload["evidence"], self.documents[:4])
        approve.assert_not_called()
        json.dumps(result)

    def test_six_claim_limit_is_validated_at_runtime(self):
        for count in (6, 7):
            batch = candidate.ClaimBatch.model_construct(claims=[
                candidate.AtomicClaim(claim=f"Independent project claim {i}.", source_evidence_id="e1")
                for i in range(count)])
            self.client.responses.parse.return_value = response(batch)
            documents = candidate._load_candidate_evidence("c1")
            if count == 6:
                self.assertEqual(len(candidate._extract_claims("c1", documents)), 6)
            else:
                with self.assertRaises(ProviderError):
                    candidate._extract_claims("c1", documents)

    def test_empty_extraction_is_an_empty_profile_without_checker(self):
        self.client.responses.parse.return_value = response(self.batch([]))
        result = self.call("build_candidate_profile", {"candidate_id": "c1"})
        self.assertEqual(result, {"success": True, "result": {"candidate_id": "c1", "assessments": []}})
        self.assertEqual(self.client.responses.parse.call_count, 1)
        self.rerank.assert_not_called()

    def test_extraction_rejects_missing_fabricated_and_cross_candidate_sources(self):
        for source in ("missing", "other", ""):
            batch = candidate.ClaimBatch.model_construct(claims=[
                candidate.AtomicClaim.model_construct(claim="I built an API.", source_evidence_id=source)])
            self.client.responses.parse.return_value = response(batch)
            result = self.call("build_candidate_profile", {"candidate_id": "c1"})
            self.assertFalse(result["success"])
            self.assertIn(result["error_type"], ("ToolError", "ProviderError"))
        batch = candidate.ClaimBatch.model_construct(claims=[candidate.AtomicClaim.model_construct(claim="API")])
        self.client.responses.parse.return_value = response(batch)
        self.assertEqual(self.call("build_candidate_profile", {"candidate_id": "c1"})["error_type"], "ProviderError")
        self.rerank.assert_not_called()

    def test_extraction_rejects_duplicate_and_empty_claims(self):
        for claims in ([("I built an API.", "e1"), ("i built an api!", "e3")], [("   ", "e1")]):
            self.client.responses.parse.return_value = response(self.batch(claims))
            self.assertEqual(self.call("build_candidate_profile", {"candidate_id": "c1"})["error_type"], "ToolError")

    def test_single_claim_bypasses_extraction_and_preserves_original_claim(self):
        self.client.responses.parse.return_value = response(self.assessment())
        claim = "I built a Django API for a university project."
        with patch.object(candidate, "_extract_claims") as extract:
            result = self.call(arguments={"candidate_id": "c1", "claim": claim})
        extract.assert_not_called()
        self.assertEqual(result["result"]["claim"], claim)
        self.assertEqual(self.client.responses.parse.call_count, 1)
        self.assertNotIn("source_evidence_id", result["result"])
        self.assertNotIn("is_atomic", result["result"])
        json.dumps(result)

    def test_parent_recovery_deduplicates_ranked_ids_and_preserves_metadata(self):
        documents = candidate._load_candidate_evidence("c1")
        self.client.responses.parse.return_value = response(self.assessment())
        with patch.object(candidate, "dense_search", return_value=self.chunks("e1", "e1", "e3")), \
                patch.object(candidate, "bm25_search", return_value=self.chunks("e1", "e2")):
            ids = candidate._retrieve_evidence_ids("c1", "Django API", documents)
        self.assertEqual(ids, ["e1", "e3", "e2"])
        result = candidate._check_claim("c1", "Django API", ids, documents)
        payload = json.loads(self.client.responses.parse.call_args.kwargs["input"][1]["content"])
        self.assertEqual(payload["evidence"], [self.documents[i] for i in (0, 2, 1)])
        citation = result["selected_evidence"][0]
        self.assertEqual({key: citation[key] for key in ("date", "source_id", "category")},
                         {key: self.documents[0][key] for key in ("date", "source_id", "category")})
        self.assertNotIn("text", citation)

    def test_origin_is_preserved_when_retrieval_omits_it(self):
        self.client.responses.parse.side_effect = [response(self.batch()), response(self.assessment())]
        with patch.object(candidate, "dense_search", return_value=self.chunks("e3")), \
                patch.object(candidate, "bm25_search", return_value=self.chunks("e4")):
            result = self.call("build_candidate_profile", {"candidate_id": "c1"})
        self.assertTrue(result["success"])
        payload = json.loads(self.client.responses.parse.call_args.kwargs["input"][1]["content"])
        self.assertEqual([d["evidence_id"] for d in payload["evidence"]], ["e3", "e4", "e1"])

    def test_independent_chunk_top_k_and_literal_keywords(self):
        claim = "I built a Django REST API using PostgreSQL in C++ and .NET."
        documents = candidate._load_candidate_evidence("c1")
        indexed = database.load_indexed_chunks(candidate.EMBEDDING_MODEL, "c1")
        with patch.object(candidate, "dense_search", return_value=self.chunks("e1", "e2")) as dense, \
                patch.object(candidate, "bm25_search", return_value=self.chunks("e3", "e4")) as bm25:
            ids = candidate._retrieve_evidence_ids("c1", claim, documents)
        self.assertEqual(ids, ["e1", "e2", "e3", "e4"])
        dense.assert_called_once_with(claim, indexed, top_k=20)
        self.assertEqual(bm25.call_count, 1)
        request = bm25.call_args.kwargs
        self.assertEqual(request["candidate_id"], "c1")
        self.assertEqual(request["top_k"], 20)
        keywords = bm25.call_args.args[0]
        self.assertTrue(all(word in claim for word in keywords))
        self.assertIn("PostgreSQL", keywords)
        self.assertIn("C++", keywords)
        self.assertIn(".NET", keywords)

    def test_merge_preserves_first_occurrence_and_deduplicates_across_searches(self):
        documents = candidate._load_candidate_evidence("c1")
        with patch.object(candidate, "RETRIEVAL_K", 5), \
                patch.object(candidate, "dense_search", return_value=self.chunks("e1", "e1", "e1", "e2", "e1")) as dense, \
                patch.object(candidate, "bm25_search", return_value=self.chunks("e1", "e3", "e3", "e1", "e4")) as bm25:
            ids = candidate._retrieve_evidence_ids("c1", "API", documents)
        self.assertEqual(ids, ["e1", "e2", "e3", "e4"])
        dense.assert_called_once()
        bm25.assert_called_once()
        self.assertEqual(dense.call_args.kwargs["top_k"], 5)
        self.assertEqual(bm25.call_args.kwargs["top_k"], 5)

    def test_one_parent_is_valid_without_refilling_chunk_quota(self):
        self.client.responses.parse.return_value = response(self.assessment())
        with patch.object(candidate, "RETRIEVAL_K", 5), \
                patch.object(candidate, "dense_search", return_value=self.chunks("e1", "e1", "e1", "e1", "e1")) as dense, \
                patch.object(candidate, "bm25_search", return_value=[]) as bm25:
            result = self.call()
        self.assertTrue(result["success"])
        payload = json.loads(self.client.responses.parse.call_args.kwargs["input"][1]["content"])
        self.assertEqual(payload["evidence"], [self.documents[0]])
        dense.assert_called_once()
        bm25.assert_called_once()
        self.assertEqual(dense.call_args.kwargs["top_k"], 5)
        self.assertEqual(bm25.call_args.kwargs["top_k"], 5)

    def test_no_second_parent_top_k_is_applied(self):
        documents = candidate._load_candidate_evidence("c1")
        with patch.object(candidate, "RETRIEVAL_K", 2), \
                patch.object(candidate, "dense_search", return_value=self.chunks("e1", "e2")) as dense, \
                patch.object(candidate, "bm25_search", return_value=self.chunks("e3", "e4")) as bm25:
            ids = candidate._retrieve_evidence_ids("c1", "API", documents)
        self.assertEqual(ids, ["e1", "e2", "e3", "e4"])
        dense.assert_called_once()
        bm25.assert_called_once()
        self.assertEqual(dense.call_args.kwargs["top_k"], 2)
        self.assertEqual(bm25.call_args.kwargs["top_k"], 2)

    def test_empty_search_results_do_not_drop_the_other_methods_results(self):
        documents = candidate._load_candidate_evidence("c1")
        for dense_ids, bm25_ids, expected in ((["e3", "e1"], [], ["e3", "e1"]),
                                             ([], ["e2"], ["e2"])):
            with self.subTest(dense=dense_ids, bm25=bm25_ids), \
                    patch.object(candidate, "dense_search", return_value=self.chunks(*dense_ids)) as dense, \
                    patch.object(candidate, "bm25_search", return_value=self.chunks(*bm25_ids)) as bm25:
                self.assertEqual(candidate._retrieve_evidence_ids("c1", "API", documents), expected)
            dense.assert_called_once()
            bm25.assert_called_once()

    def test_real_dense_and_bm25_searches_are_candidate_scoped(self):
        documents = candidate._load_candidate_evidence("c1")
        with patch.object(candidate, "dense_search", wraps=candidate.dense_search) as dense, \
                patch.object(candidate, "bm25_search", wraps=candidate.bm25_search) as bm25:
            ids = candidate._retrieve_evidence_ids("c1", "Django REST API", documents)
        self.assertIn("e1", ids)
        self.assertNotIn("other", ids)
        self.assertTrue(all(c["candidate_id"] == "c1" for c in dense.call_args.args[1]))
        self.assertEqual(bm25.call_args.kwargs["candidate_id"], "c1")
        self.embed.assert_called_once_with(["Django REST API"])

    def test_both_workflows_use_no_rrf_or_reranker(self):
        self.client.responses.parse.side_effect = [response(self.assessment()),
            response(self.batch()), response(self.assessment())]
        with patch.object(reranker, "OpenAI") as rerank_client:
            self.assertTrue(self.call()["success"])
            self.assertTrue(self.call("build_candidate_profile", {"candidate_id": "c1"})["success"])
        self.rerank.assert_not_called()
        self.fusion.assert_not_called()
        rerank_client.assert_not_called()
        self.assertEqual(self.client.responses.parse.call_count, 3)

    def test_invalid_retrieved_or_recovered_ids_fail_explicitly(self):
        documents = candidate._load_candidate_evidence("c1")
        for chunk in (dict(candidate_id="c2", evidence_id="other"),
                      dict(candidate_id="c1", evidence_id="missing")):
            with patch.object(candidate, "dense_search", return_value=[chunk]), \
                    patch.object(candidate, "bm25_search", return_value=[]):
                with self.assertRaises(ToolError):
                    candidate._retrieve_evidence_ids("c1", "API", documents)
        for evidence_id in ("missing", "other"):
            with self.assertRaises(ToolError):
                candidate._check_claim("c1", "API", [evidence_id], documents)
        documents["e1"] = {**documents["e1"], "candidate_id": "c2"}
        with self.assertRaises(ToolError):
            candidate._check_claim("c1", "API", ["e1"], documents)
        self.client.responses.parse.assert_not_called()

    def test_cross_candidate_or_missing_bm25_items_are_rejected(self):
        documents = candidate._load_candidate_evidence("c1")
        for item in (dict(candidate_id="c2", evidence_id="other"),
                     dict(candidate_id="c2", evidence_id="e1"),
                     dict(candidate_id="c1", evidence_id="missing")):
            with patch.object(candidate, "dense_search", return_value=self.chunks("e1")), \
                    patch.object(candidate, "bm25_search", return_value=[item]), \
                    self.assertRaises(ToolError):
                candidate._retrieve_evidence_ids("c1", "API", documents)
        self.client.responses.parse.assert_not_called()

    def test_wrong_candidate_storage_records_are_rejected(self):
        with patch.object(candidate, "get_evidence_documents", return_value=[self.documents[-1]]):
            with self.assertRaises(ToolError):
                candidate._load_candidate_evidence("c1")

    def test_missing_partial_and_outdated_indexes_are_errors(self):
        documents = candidate._load_candidate_evidence("c1")
        indexed = database.load_indexed_chunks(candidate.EMBEDDING_MODEL, "c1")
        for records in ([], indexed[:-1], [{**c, "text": "outdated text"} for c in indexed]):
            with patch.object(candidate, "load_indexed_chunks", return_value=records):
                with self.assertRaisesRegex(ToolError, "index"):
                    candidate._retrieve_evidence_ids("c1", "API", documents)
        self.embed.assert_not_called()

    def test_cross_candidate_index_and_bm25_storage_failures_are_errors(self):
        documents = candidate._load_candidate_evidence("c1")
        indexed = database.load_indexed_chunks(candidate.EMBEDDING_MODEL, "c1")
        indexed[0]["candidate_id"] = "c2"
        with patch.object(candidate, "load_indexed_chunks", return_value=indexed):
            with self.assertRaisesRegex(ToolError, "another candidate"):
                candidate._retrieve_evidence_ids("c1", "API", documents)
        with patch.object(candidate, "bm25_search", side_effect=StorageError("BM25 unavailable")):
            self.assertEqual(self.call()["error_type"], "StorageError")
        self.client.responses.parse.assert_not_called()

    def test_supported_contradicted_and_insufficient_results_are_distinct(self):
        cases = [("I built a Django API for a university project.", self.assessment()),
            ("I made the stated production deployment.", self.assessment("contradicted", [
                dict(evidence_id="e4", role="contradicts", quote=self.documents[3]["text"])])),
            ("I was employed at NASA.", self.assessment("insufficient_evidence", [],
                reason="No supplied record establishes the asserted employment."))]
        for claim, assessment in cases:
            self.client.responses.parse.return_value = response(assessment)
            result = self.call(arguments={"candidate_id": "c1", "claim": claim})
            self.assertTrue(result["success"])
            self.assertEqual(result["result"]["verdict"], assessment.verdict)

    def test_successful_empty_retrieval_can_produce_insufficient_evidence(self):
        self.client.responses.parse.return_value = response(self.assessment("insufficient_evidence", []))
        with patch.object(candidate, "dense_search", return_value=[]), \
                patch.object(candidate, "bm25_search", return_value=[]):
            result = self.call()
        self.assertEqual(result["result"]["verdict"], "insufficient_evidence")
        payload = json.loads(self.client.responses.parse.call_args.kwargs["input"][1]["content"])
        self.assertEqual(payload["evidence"], [])

    def test_temporal_change_keeps_both_sources_and_preserves_clear_support(self):
        self.client.responses.parse.return_value = response(self.assessment(citations=[
            dict(evidence_id="e3", role="supports", quote=self.documents[2]["text"]),
            dict(evidence_id="e2", role="qualifies", quote=self.documents[1]["text"])],
            reason="The 2026 SQL project supersedes the 2024 lack of experience; both historical records remain valid."))
        result = self.call(arguments={"candidate_id": "c1", "claim": "I completed an SQL project by 2026."})
        self.assertEqual(result["result"]["verdict"], "supported")
        self.assertEqual([c["date"] for c in result["result"]["selected_evidence"]], ["2026-06", "2024-01"])

    def test_complementary_and_conflicting_citations_are_retained(self):
        self.client.responses.parse.return_value = response(self.assessment(citations=[
            self.assessment().selected_evidence[0].model_dump(),
            dict(evidence_id="e3", role="supports", quote=self.documents[2]["text"])]))
        result = self.call(arguments={"candidate_id": "c1", "claim": "I completed backend development projects."})
        self.assertEqual(len(result["result"]["selected_evidence"]), 2)
        documents = candidate._load_candidate_evidence("c1")
        documents["e1"]["text"] = "The candidate deployed the Django API to production in July 2026."
        self.client.responses.parse.return_value = response(self.assessment("insufficient_evidence", [
            dict(evidence_id="e1", role="supports", quote=documents["e1"]["text"]),
            dict(evidence_id="e4", role="contradicts", quote=documents["e4"]["text"])],
            reason="The supplied accounts materially conflict about the same deployment."))
        result = candidate._check_claim("c1", "I deployed the Django API to production in July 2026.",
                                        ["e1", "e4"], documents)
        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertEqual([c["role"] for c in result["selected_evidence"]], ["supports", "contradicts"])

    def test_quotes_must_be_exact_nonempty_substrings(self):
        for quote in ("Built a REST API", "built a REST API", "Built an imaginary API", "  "):
            self.client.responses.parse.return_value = response(self.assessment(citations=[
                dict(evidence_id="e1", role="supports", quote=quote)]))
            result = self.call()
            self.assertEqual(result["success"], quote == "Built a REST API")
        self.assertNotEqual("Built a REST API", self.documents[0]["text"])

    def test_fabricated_foreign_and_out_of_context_citations_are_rejected(self):
        for evidence_id in ("missing", "other", "e2"):
            self.client.responses.parse.return_value = response(self.assessment(citations=[
                dict(evidence_id=evidence_id, role="supports", quote=self.documents[1]["text"])]))
            with patch.object(candidate, "_retrieve_evidence_ids", return_value=["e1"]):
                result = self.call()
            self.assertFalse(result["success"])
            self.assertEqual(result["error_type"], "ToolError")

    def test_duplicate_citations_and_more_than_two_selections_are_rejected(self):
        citation = self.assessment().selected_evidence[0]
        self.client.responses.parse.return_value = response(self.assessment(citations=[
            citation.model_dump(), {**citation.model_dump(), "quote": "Built a REST API"}]))
        self.assertEqual(self.call()["error_type"], "ToolError")
        excessive = candidate.ClaimAssessment.model_construct(
            is_atomic=True, verdict="supported", reason="Too many quotes", selected_evidence=[citation] * 3)
        self.client.responses.parse.return_value = response(excessive)
        self.assertEqual(self.call()["error_type"], "ProviderError")

    def test_identical_quotes_from_different_evidence_are_rejected(self):
        documents = candidate._load_candidate_evidence("c1")
        documents["e3"]["text"] += "\nBuilt a REST API"
        self.client.responses.parse.return_value = response(self.assessment(citations=[
            dict(evidence_id="e1", role="supports", quote="Built a REST API"),
            dict(evidence_id="e3", role="supports", quote="Built a REST API")]))
        with self.assertRaisesRegex(ToolError, "Duplicate"):
            candidate._check_claim("c1", "API", ["e1", "e3"], documents)

    def test_injected_metadata_invalid_verdict_and_nonatomic_claim_are_errors(self):
        assessment = self.assessment()
        invalid = assessment.model_dump()
        invalid["selected_evidence"][0]["date"] = "invented"
        with self.assertRaises(ValidationError):
            candidate.ClaimAssessment.model_validate(invalid)
        for verdict in ("unsupported", "uncertain"):
            invalid = {**assessment.model_dump(), "verdict": verdict}
            with self.assertRaises(ValidationError):
                candidate.ClaimAssessment.model_validate(invalid)
        self.client.responses.parse.return_value = response(self.assessment(is_atomic=False))
        self.assertEqual(self.call()["error_type"], "ToolError")

    def test_decisive_verdict_requires_corresponding_quotation(self):
        for verdict in ("supported", "contradicted"):
            self.client.responses.parse.return_value = response(self.assessment(verdict, []))
            self.assertEqual(self.call()["error_type"], "ToolError")

    def test_both_stages_use_fixed_model_effort_and_shared_rubric(self):
        self.client.responses.parse.side_effect = [response(self.batch()), response(self.assessment())]
        result = self.call("build_candidate_profile", {"candidate_id": "c1"})
        self.assertTrue(result["success"])
        formats = []
        for call in self.client.responses.parse.call_args_list:
            request = call.kwargs
            self.assertEqual(request["model"], "gpt-6-luna")
            self.assertEqual(request["reasoning"], {"effort": "none"})
            self.assertFalse(request["store"])
            formats.append(request["text_format"])
            prompt = request["input"][0]["content"]
            self.assertIn(candidate.VERIFICATION_RULES, prompt)
            for instruction in ("chronology", "same applicable time scope", "not itself a contradiction",
                                "Do not default to uncertainty", "untrusted data", "production"):
                self.assertIn(instruction, prompt)
        self.assertEqual(formats, [candidate.ClaimBatch, candidate.ClaimAssessment])
        candidate.OpenAI.assert_called_with(timeout=30, max_retries=1)

    def test_context_overflow_fails_before_api_without_truncation(self):
        with patch.object(candidate, "MAX_INPUT_TOKENS", 10):
            result = self.call("build_candidate_profile", {"candidate_id": "c1"})
        self.assertEqual(result["error_type"], "ToolError")
        self.assertIn("no evidence was truncated", result["error_msg"])
        self.client.responses.parse.assert_not_called()
        self.embed.assert_not_called()

    def test_checker_context_overflow_is_an_error(self):
        documents = candidate._load_candidate_evidence("c1")
        with patch.object(candidate, "MAX_INPUT_TOKENS", 10):
            with self.assertRaises(ToolError):
                candidate._check_claim("c1", "API", ["e1"], documents)
        self.client.responses.parse.assert_not_called()

    def test_api_failures_remain_provider_errors_for_both_stages(self):
        self.client.responses.parse.side_effect = APIConnectionError(request=Mock())
        for name in ("build_candidate_profile", "verify_candidate_claim"):
            arguments = {"candidate_id": "c1", **({"claim": "API"} if name.startswith("verify") else {})}
            self.assertEqual(self.call(name, arguments)["error_type"], "ProviderError")

    def test_invalid_incomplete_or_missing_model_outputs_are_provider_errors(self):
        for parsed in (None, self.assessment()):
            self.client.responses.parse.return_value = SimpleNamespace(status="incomplete", output_parsed=parsed)
            self.assertEqual(self.call()["error_type"], "ProviderError")
        self.client.responses.parse.return_value = response(None)
        self.assertEqual(self.call()["error_type"], "ProviderError")
        self.client.responses.parse.return_value = response({"verdict": "supported"})
        self.assertEqual(self.call()["error_type"], "ProviderError")
        self.client.responses.parse.return_value = response(candidate.ClaimAssessment.model_construct(
            is_atomic=True, verdict="uncertain", reason="invalid", selected_evidence=[]))
        self.assertEqual(self.call()["error_type"], "ProviderError")

    def test_missing_api_key_and_embedding_failures_are_real_errors(self):
        with patch.dict(candidate.os.environ, {"OPENAI_API_KEY": ""}):
            self.assertEqual(self.call("build_candidate_profile", {"candidate_id": "c1"})["error_type"], "ProviderError")
        self.embed.side_effect = ProviderError("Embedding unavailable")
        self.assertEqual(self.call()["error_type"], "ProviderError")

    def test_database_failures_are_storage_errors_and_no_business_writes_occur(self):
        before = self.db_path.read_bytes()
        self.client.responses.parse.return_value = response(self.assessment())
        self.assertTrue(self.call()["success"])
        self.assertEqual(self.db_path.read_bytes(), before)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("DROP TABLE evidence")
        result = self.call()
        self.assertEqual(result["error_type"], "StorageError")
        self.assertNotEqual(result.get("verdict"), "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
