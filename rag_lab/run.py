[
  {
    "evidence_id": "ev_backend_001",
    "candidate_id": "candidate_001",
    "source_id": "health_alerts_backend",
    "category": "project",
    "text": "The candidate worked on the backend of a team health-alert application built with Django and Django REST Framework. Their responsibility focused on the AI subsystem and the retrieval of health alerts relevant to a user's date and location. They integrated external Gemini services with the application's shared alert data model and REST API. During development, they investigated a bug in geographic filtering where prefix-based location matching returned incorrect results for some regions. They traced the problem through the retrieval path and corrected the matching logic. The candidate also implemented persistent caching for disease metadata and country-level risk information. The cache distinguished between new, expired, valid, and failed entries and supported refresh behavior when stored information became stale. Validation was added around external service results before data entered the rest of the application. This project gave the candidate practical experience with REST APIs, backend debugging, caching, external AI service integration, validation, and data retrieval in a multi-developer codebase."
  },
  {
    "evidence_id": "ev_ml_001",
    "candidate_id": "candidate_001",
    "source_id": "synthetic_face_research",
    "category": "research",
    "text": "The candidate conducted an honours research project on generalizable detection of AI-generated face images. They configured several released synthetic-image detector checkpoints and prepared image generation pipelines covering both GAN-based and diffusion-based generators. The evaluation compared detector performance when training and testing conditions were similar as well as when detectors encountered images from different generators. Performance was measured using metrics including area under the ROC curve and average precision. The candidate also investigated robustness under image transformations such as JPEG compression, cropping, resizing, and changes to StyleGAN3 truncation. Experiments used controlled sets of real and synthetic face images for each generator configuration. The work required managing PyTorch-based inference pipelines, reproducing existing research implementations, designing benchmark conditions, recording experimental limitations, and interpreting differences between same-generator and cross-generator performance. The project developed practical experience in computer vision, machine learning evaluation, generative AI, reproducible experimentation, and robustness analysis."
  },
  {
    "evidence_id": "ev_simulation_001",
    "candidate_id": "candidate_001",
    "source_id": "capacity_planning_simulation",
    "category": "project",
    "text": "The candidate developed a Python discrete-event simulator for capacity planning in a system containing two server groups. Incoming jobs entered FIFO queues and could be redirected between server groups according to the simulated system rules. The simulator supported both trace-driven inputs and seeded stochastic workloads so that experiments could be reproduced. To study resource allocation, the candidate evaluated eleven different ways of dividing twelve servers between the two groups. Each allocation was tested through repeated simulation runs. Common random numbers were used across configurations to reduce unnecessary experimental variance, and paired 95 percent confidence intervals were used when comparing performance. Under the modelled workload, an even six-six allocation produced the strongest result. The candidate also validated generated service-time and inter-arrival samples against their intended probability distributions and fixed random seeds to make experiments reproducible. The work involved queueing behaviour, stochastic modelling, experimental design, statistical comparison, Python implementation, debugging, and performance-oriented systems analysis."
  }
]