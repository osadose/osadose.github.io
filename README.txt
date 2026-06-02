AWS SageMaker Unified Studio

Application to Polaris HPI

Slide 1 — Title

AWS SageMaker Unified Studio
Immersion Day Findings & Application to Polaris HPI

Presented by: Osama
Date: June 2026

Slide 2 — What Is It

SageMaker Unified Studio is not a replacement — it is a targeted addition to our data science layer

Four capabilities in one platform:



|Component          |What It Does                   |
|-------------------|-------------------------------|
|Amazon DataZone    |Data cataloguing and governance|
|Processing Jobs    |Scalable compute for pipelines |
|SageMaker Pipelines|Workflow orchestration         |
|Studio Spaces      |Cloud development environments |

VS Code Copilot confirmed by reading our actual repo:

“This is primarily a Django data-catalogue app on ECS/S3. SageMaker can be added for data science workflows without replacing the Django/ECS app.”

Integration boundary: S3
SageMaker writes outputs to the same S3 paths the catalogue already reads from. No changes to the web app needed.

Slide 3 — Where It Fits in Our Stack

A clean separation of concerns:

|Keep As-Is               |Add SageMaker               |
|-------------------------|----------------------------|
|Django UI and API        |HPI modelling pipeline runs |
|ECS deployment           |Notebook-based investigation|
|Cognito and auth         |Reproducible batch jobs     |
|Standard asset publishing|Monthly scheduled HPI runs  |
|Simple chart rendering   |Experiment tracking and QA  |

The practical split:

┌─────────────────────────────────────────────┐
│           KEEP AS-IS (ECS)                  │
│  Django UI → Cognito Auth → Asset Publishing│
└─────────────────────┬───────────────────────┘
                      │
                   S3 Boundary
                      │
┌─────────────────────▼───────────────────────┐
│         ADD SAGEMAKER                        │
│  Processing Jobs → Pipelines → Studio Spaces │
│  HPI Modelling → QA → DataZone Governance   │
└─────────────────────────────────────────────┘


Slide 4 — The Strongest Case: HPI Modelling

Copilot found concrete evidence in the repo:

	•	modelling.py and imputation.py already use statsmodels and scikit-learn
	•	acorn.py and energy_performance_certs.py show the S3 ingestion pattern
	•	io.py already has the S3 storage helpers that become the integration point

We are not rewriting anything — just moving where the code runs

Current Lambda approach:

# Capped at 15 min, 10GB memory
# Breaks for England & Wales volumes
# Tightly coupled to DAC2
def data_getter(
    *, logger, api_client, ...
) -> PipelineFunctionReturnType:
    return None, file_object


SageMaker Processing Job:

# No caps — scales to England & Wales
# Same HPI library code unchanged
# Reads from S3, writes to S3
def main():
    df = hpi.data_access.ros.from_raw(input_path)
    df.to_parquet(output_path)


The HPI library code does not change at all.

Slide 5 — Four Problems It Solves Right Now

Grounded in what Copilot confirmed and what the immersion day demonstrated:

1. Compute limits blocking England & Wales
Processing Jobs scale freely — no Lambda caps. Same code, bigger instance type.

2. No processing flow visibility
SageMaker Pipelines shows the full DAG with live step status. Business area sees run success before looking for output.

3. Local development unreliable
Studio Spaces replace Docker and LocalStack. Each team member gets their own cloud environment. What works in dev works in production.

4. Logging and observability gap
All Processing Job logs go to CloudWatch automatically. Instrument once with ADOT — send to CloudWatch now, switch to any backend later without changing code.
