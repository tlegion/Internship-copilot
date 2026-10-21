from typing import Annotated

from pydantic import Field

from storages.database import (
    get_job, 
    get_all_jobs, 
    check_exists, 
    get_last_job, 
    insert_job,
    check_similar,
    get_jobs_page,
)
import re
import sqlite3
from errors import ToolError

# from pathlib import Path
# ROOT = Path(__file__).parent[0]
# import sys
# sys.path.insert(0, ROOT)
import approval

# JOBS = {
#     "job_001": {
#         "title": "AI Engineering Intern",
#         "requirements": ["Python", "API integration", "SQL"]
#     }
# }



def lookup_job(job_id:str):

    """Retrieve job details by job ID.

    Args:
        job_id: Job ID, for example job_001
    """
    job = get_job(job_id)
    if job is None:
        raise ToolError(f"Job not found: {job_id}")
    return job

def list_jobs(
    limit: Annotated[int, Field(ge=1, le=50)] | None = 20,
    after_job_id: str | None = None,
    company: str | None = None,
    location: str | None = None,
) -> dict:
    """Browse stored jobs without knowing their IDs in advance; use lookup_job for one known ID.
    Paginate in ascending order of the numeric job ID suffix. When has_more is true, pass
    next_cursor as after_job_id on the next call and keep the same filters. company and location are
    exact filters that can be used independently or together; pass null to disable a filter.

    Args:
        limit: Number of jobs per page, 1 to 50; null defaults to 20.
        after_job_id: Return jobs with a numeric suffix greater than this ID, for example
            job_010; null starts from the beginning. The ID does not need to exist.
        company: Exact company name filter; null disables it.
        location: Exact location filter; null disables it.
    """
    if limit is None:
        limit = 20
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ToolError("limit must be a positive integer no greater than 50")
    if after_job_id is not None and (
        not isinstance(after_job_id, str)
        or re.fullmatch(r"job_[0-9]+", after_job_id) is None
    ):
        raise ToolError("after_job_id must have the format job_<digits>")
    for name, value in (("company", company), ("location", location)):
        if value is not None and not isinstance(value, str):
            raise ToolError(f"{name} must be a string or null")
    return get_jobs_page(limit, after_job_id, company, location)


def search_jobs_hybrid(
    query: str,
    keywords: list[str],
    company: str | None = None,
    location: str | None = None,
    top_k: Annotated[int, Field(ge=1, le=20)] | None = 10,
) -> dict:
    """Find stored jobs by natural-language role preferences, technologies and skills. Use meaningful
    English semantic queries for English job records and distinctive technical keywords for lexical
    retrieval. Rank is retrieval relevance, not candidate fit. Use list_jobs for browsing with exact
    filters and pagination; lookup_job for one known ID. Company and location are optional exact
    filters. An outdated index must be synchronized by the developer before searching.

    Args:
        query: Nonempty English semantic search query.
        keywords: Distinctive lexical anchors such as Python, RAG, LLM APIs; [] is allowed.
        company: Exact company filter; null disables it.
        location: Exact location filter; null disables it.
        top_k: Number of distinct jobs, 1 to 20; null defaults to 10.
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a nonempty natural-language string")
    if not isinstance(keywords, list) or any(not isinstance(word, str) for word in keywords):
        raise ToolError("keywords must be a list of strings")
    for name, value in (("company", company), ("location", location)):
        if value is not None and not isinstance(value, str):
            raise ToolError(f"{name} must be a string or null")
    if top_k is None:
        top_k = 10
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ToolError("top_k must be a positive integer no greater than 20")
    from rag_lab.job_hybrid_search import hybrid_job_search
    return hybrid_job_search(query, keywords, company, location, top_k)


def new_job_id():
    last = get_last_job()
    if last:
        last_id = re.search(r"\d+", last)
        last_id_int = int(last_id.group())
        new_id_int = last_id_int + 1
        return f"job_{new_id_int:03d}"
    else:
        raise ToolError("No job exists")
    


def create_job(
        title: str,
        company: str,
        description: str,
        location: str | None,
) -> dict:
    """Create a new job record from job information provided by the user. Call only when the job title,
    company, and job description can be clearly extracted from the input. Do not generate job_id
    yourself; the system generates it internally. If no location is provided, pass null for
    location.

    Args:
        title: Job title, for example Software Engineering Intern
        company: Company name
        description: Job responsibilities or description; use only information provided by the
            user and do not add requirements that were not provided
        location: Job location; pass null if the original information does not explicitly
            provide it
    """
    if check_exists(title, company, description, location):
        raise ToolError("job already exists")
    similars = check_similar(title, company, description, location)
    if similars:
        pending_job = {
            "title": title,
            "company": company,
            "description": description,
            "location": location,
        }
        similar_job_ids = [job["id"] for job in similars]
        if not approval.review_similar_job_creation("create_job", pending_job, similars):
            return {
                "status": "cancelled",
                "job_id": None,
                "similar_job_ids": similar_job_ids,
                "user_selection": False,
            }

    new_id = new_job_id()
    
    result = {
        **insert_job(new_id, title, company, description, location),
        "status": "created",
        "job_id": new_id,
    }
    if similars:
        result.update(similar_job_ids=similar_job_ids, user_selection=True)
    return result
    
    
    
