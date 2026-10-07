from storages.database import (
    get_job, 
    get_all_jobs, 
    check_exists, 
    get_last_job, 
    insert_job
)
import re
import sqlite3
from errors import ToolError

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
    new_id = new_job_id()
    
    return insert_job(new_id,title,company, description, location)
    
    
    
