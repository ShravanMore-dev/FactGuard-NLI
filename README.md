# FactGuard-NLI: Sentence-Level NLI Faithfulness & Hallucination Guardrail

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![ROCm 7.14](https://img.shields.io/badge/AMD_ROCm-7.14-red.svg)](https://rocm.docs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ROCm_Enabled-ee4c2c.svg)](https://pytorch.org/)

**VeriRAG** is an offline enterprise hallucination guardrail and verification system designed for Retrieval-Augmented Generation (RAG) pipelines in high-stakes domains (legal, medical, defense, and technical documentation). 

It replaces slow, non-deterministic LLM-as-a-Judge self-evaluators with a fast local cross-encoder model (`cross-encoder/nli-deberta-v3-large`). The system performs sentence-by-sentence Natural Language Inference (NLI) checks on LLM-generated responses against retrieved source context, providing real-time factual badges (**Supported**, **Unverified**, **Contradiction**) and triggering automated single-pass correction loops for conflicting claims.

This codebase is hardware-optimized for **AMD GPUs** using **ROCm 7.14** within a **16 GB VRAM budget**.

---

## Key Features

- **Real-Time Factual Badging:** Labels every generated claim as **Supported**, **External Knowledge**, or **Contradiction**.
- **Deterministic & Non-LLM Verification:** Eliminates LLM self-bias using a dedicated high-precision DeBERTa-v3 NLI cross-encoder.
- **Automated Self-Correction:** Intercepts claims flagged as contradictions and executes a targeted re-prompting pass before presenting the response.
- **100% Offline & Private:** Zero external API dependencies; all models run locally on AMD ROCm hardware stack.
- **AMD Hardware Acceleration:** Optimized for ROCm 7.14 using HIP-backed PyTorch and vLLM / llama-cpp execution engines.

---

## Architecture & Workflow

```text
[ User Query ] ──► [ Local Vector Search ] ──► [ Top-K Context Chunks ]
                                                       │
                                                       ▼
[ Verified Answer + Badges ] ◄── [ NLI Verifier Engine ] ◄── [ Local LLM Generation ]
