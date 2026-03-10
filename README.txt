---
title: "Week 10: LLMs in Practice"
date: last-modified
format:
  revealjs:
    theme:
      - default
    slide-number: true
    auto-stretch: false
    logo: ../../images/logo_colour.png
    embed-resources: true
scrollable: true
---

## Welcome to Week 10 {.smaller}

**Theme:** LLMs in practice

**Goals:**

  - Understand what ClassifAI is and where it helps in official statistics work

  - Run a simple end-to-end text classification example in a notebook

  - Evaluate model outputs and give structured feedback on the demo

---

## Why ClassifAI? {.smaller}

- Manual text labelling is slow, inconsistent, and hard to scale.

- ClassifAI helps teams categorise large volumes of text consistently.

- Useful examples: survey comments, complaints, open-ended responses, and helpdesk tickets.

- Today we focus on practical usage, not deep theory.

---

## ClassifAI Workflow {.smaller}

1. Define categories and success criteria
2. Load and clean labelled text data
3. Convert text into model features/embeddings
4. Train or run a classifier
5. Predict class labels for new text
6. Evaluate results and inspect errors
7. Refine data, labels, and model settings

---

## Quick Notebook Demo {.smaller}

- Input: short text records and target class labels

- Process: preprocess text, fit classifier, generate predictions

- Output: predicted class + confidence score per record

- We will compare a few predictions against true labels live

---

## How We Measure Quality {.smaller}

- **Accuracy:** overall correct predictions
- **Precision:** how often predicted positives are correct
- **Recall:** how many true positives we captured
- **Confusion matrix:** where the model confuses categories

Accuracy alone can hide poor performance for minority classes.

---

## Let’s Dive Into The Live Session {.center style="text-align: center;"}

