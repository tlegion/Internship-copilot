---
name: job-comparison
description: Compare internship jobs by their stated roles, technologies, responsibilities and locations. Use when the user wants to compare or choose between stored jobs.
metadata:
  version: "1.0"
---

# Compare internship jobs

1. Identify the user's priorities and any exact company or location restrictions.
2. Use `search_jobs_hybrid` for natural-language preferences, `list_jobs` for
   browsing with pagination, or `lookup_job` for known IDs.
3. Read [the comparison guide](references/comparison-guide.md) with
   `read_skill_resource` when preparing a comparison.
4. Compare the original job records. Separate explicit requirements from your
   interpretation; mark missing salary, experience or schedule information as
   unknown. Do not invent qualifications or infer candidate-job fit from retrieval rank.
5. Give a concise comparison and explain how each option relates to the user's
   stated priorities. Include Job IDs so the user can inspect the records.

Use only tools available in the current request. This Skill supplies instructions,
not authorization to execute scripts or bypass tool approval.
