from storages.database import get_job


# JOBS = {
#     "job_001": {
#         "title": "AI Engineering Intern",
#         "requirements": ["Python", "API integration", "SQL"]
#     }
# }



def lookup_job(input):
    job_id = input.get("job_id", "")
    if not job_id:
        return { "error_msg": f"parameter error: job_id not found"}
    job = get_job(job_id)
    if job is None:
        return {"error_msg": f"Job not found: {job_id}"}
    return job
