
Here's a ready-to-paste brief — written so Copilot has full context on both the practice dataset and why each step matters, before you start coding.

---

# Project Brief: Behavioural Segmentation Practice Pipeline (Bank Marketing Dataset)

## Context

This is a training exercise for a data science team preparing to build a behavioural population segmentation pipeline for a national census programme (not represented in this dataset). The real project will segment geographic areas by response behaviour (likelihood to respond, preferred contact channel, response to follow-up contact) and generate statistical parameters feeding a simulation model.

We cannot practice on the real (restricted, confidential) census data yet, so we're using the UCI/Kaggle **Bank Marketing dataset** as a structural stand-in. It isn't about banking — it's practice for building and testing the same kind of pipeline: segment a population, then estimate behavioural parameters for each segment.

**Important: findings from this dataset are not real insights about anything — this is purely a tooling and method rehearsal.**

## How the dataset maps to the real problem

| Bank Marketing column | Real-project equivalent | Notes |
|---|---|---|
| `y` (subscribed yes/no) | Response outcome (responded / didn't respond) | Our main outcome variable |
| `contact` (cellular/telephone) | Contact channel | Mode of contact, like online vs paper |
| `campaign` (number of contacts this campaign) | Contact attempts | How many times someone was approached |
| `pdays` (days since last contact) | Time since last intervention | Used for time-to-event / timing modelling |
| `previous` (number of prior contacts) | Prior intervention history | Whether past contact changes future behaviour |
| `poutcome` (outcome of previous campaign) | Prior response behaviour | Does past behaviour predict future behaviour |
| `age`, `job`, `marital`, `education`, `housing`, `loan` | Demographic/area covariates | Equivalent to census demographic attributes |
| `duration` | **Exclude from modelling** | This is a known data-leakage field in this dataset (call duration is only known after the call happens) — do not use as a predictor, only for post-hoc analysis if needed |

## The four stages we're practising (in order)

### Stage 1 — Variable definition
Before coding: write out explicitly which columns represent which behavioural concept (see mapping table above). Confirm which column is the outcome we're trying to explain (`y`) and which are legitimate predictors (exclude `duration`).

### Stage 2 — Segmentation
**Goal**: group records into a small number of behaviourally distinct segments using only the covariates (not the outcome `y`), then check afterwards whether the segments differ meaningfully in subscription rate.

**Methods to implement, in this order:**
1. K-Means clustering (baseline, quick)
2. Hierarchical clustering (comparison)
3. Latent Class Analysis (LCA) — harder, more realistic to the real project's likely method. Python has no mature native LCA library; try the `stepmix` package first. As a stretch goal, also implement the same analysis in R using `poLCA` (via the R extension in VSCode) and compare — we genuinely don't yet know which language/tool will work best for the real project, so this comparison is real, useful evidence, not just an exercise.

**Validation expectation**: after clustering, check that resulting segments show real differences in `y` (subscription rate) — if they don't, that's a meaningful finding about the method, not a failure to hide.

### Stage 3 — Parameter estimation per segment
**Goal**: for each segment identified in Stage 2, estimate behavioural parameters with proper uncertainty, not single point estimates.

**Methods to implement:**
- Logistic regression per segment → probability of subscribing (`y`)
- Time-to-event / survival analysis using `pdays` and `previous` → use the `lifelines` package
- Bayesian hierarchical model → estimate subscription probability per segment, allowing small segments to "borrow strength" from larger ones. Use `PyMC`. This is the hardest tool here and the one most worth spending real time on — the real project will likely depend on this.

### Stage 4 — Parameter packaging
**Goal**: output a structured, documented parameter table — not just numbers in a notebook.

**Required table schema** (practice using exactly this structure, since it mirrors what we'll need for the real project):

```
segment_id | variable_name | parameter_type | value_or_distribution | confidence_level | evidence_source | notes
```

Store this as a local SQLite table or CSV (standing in for BigQuery in the real environment).

## Engineering standards to follow throughout

- All code in a shared GitHub repo, with meaningful commit messages.
- Every notebook/script should run end-to-end from raw CSV to final parameter table — no manual steps that aren't documented.
- Write a short README per stage explaining what was tried, what worked, what didn't — we want the *process*, including failures, documented, not just a clean final notebook.
- Where a method or library choice was difficult or ambiguous, add a comment explaining why the chosen option was picked over alternatives — this becomes useful evidence for real-project tooling decisions later.

## What "done" looks like for this practice project

- A working pipeline, Stage 1 through 4, runnable end to end.
- A short comparison write-up: Python-only LCA (`stepmix`) vs. R-based LCA (`poLCA`) — which was easier to use, more reliable, better documented.
- A note on whether PyMC's Bayesian hierarchical model was straightforward to fit, and how long it took to run — this is a genuine data point for estimating real-project timelines.
- The final parameter table, in the schema above, reviewed by at least one other team member before being considered complete (practising the two-reviewer standard we'll need on the real project).