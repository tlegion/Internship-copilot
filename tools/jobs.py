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

    job = get_job(job_id)
    if job is None:
        raise ToolError(f"Job not found: {job_id}")
    return job

def list_jobs(
    limit: int = 20,
    after_job_id: str | None = None,
    company: str | None = None,
    location: str | None = None,
) -> dict:
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
    
    
    
