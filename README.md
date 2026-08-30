# QoE-Aware Self-Adaptive Hybrid Edge-Cloud Educational RAG Assistant

An educational AI assistant designed to provide personalized, knowledge-grounded, and eventually self-adaptive learning assistance across edge and cloud environments.

The system is developed incrementally through multiple phases, beginning with the core educational assistant and progressively introducing personalization, educational resource recommendation, RAG, voice interaction, QoE evaluation, experience memory, adaptive intelligence, and hybrid edge-cloud execution.

---

## Current Implementation Status

**Implemented:** Phases 1–8  
**Next planned phase:** Phase 9 - Monitoring + QoE Evaluation

| Phase | Component | Status |
|---|---|---|
| Phase 1 | Foundation and Data Layer | ✅ Implemented |
| Phase 2 | Basic Text-Based Educational Assistant | ✅ Implemented |
| Phase 3 | Learner Personalization | ✅ Implemented |
| Phase 4 | YouTube Educational Resource Recommendation | ✅ Implemented |
| Phase 5 | Knowledge Preparation + RAG + FAISS | ✅ Implemented |
| Phase 6 | CRAG / Retrieval Improvement | ✅ Implemented |
| Phase 7 | Speech-to-Text (STT) | ✅ Implemented |
| Phase 8 | Text-to-Speech (TTS) | ✅ Implemented |
| Phase 9 | Monitoring + QoE Evaluation | 🔲 Planned |
| Phase 10 | Experience Memory | 🔲 Planned |
| Phase 11 | Adaptive Intelligence | 🔲 Planned |
| Phase 12 | Hybrid Local-Cloud Routing | 🔲 Planned |
| Phase 13 | Local Model Hosting | 🔲 Planned |

---

# System Overview

The project aims to evolve from a basic educational assistant into a **QoE-aware, self-adaptive hybrid edge-cloud educational AI system**.

The overall planned architecture is:

```text
                         ┌─────────────────┐
                         │     LEARNER     │
                         └────────┬────────┘
                                  │
                           Text / Speech
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Personalization│
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ YouTube Recommendation   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Knowledge Preparation    │
                    │ Transcript → Chunks      │
                    │ → Embeddings → FAISS     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       RAG / CRAG         │
                    │ Retrieval + Evaluation   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      AI Generation       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                           AI Response
                                 │
                                 ▼
                                TTS
                                 │
                                 ▼
                             LEARNER

             ──────────── CLOSED LOOP ────────────

                    Monitoring + QoE Evaluation
                                 │
                                 ▼
                       Experience Memory
                                 │
                                 ▼
                       Adaptive Intelligence
                                 │
                                 ▼
                         Runtime Policy
                                 │
                                 ▼
                     Hybrid Edge-Cloud Routing
                         ┌───────┴───────┐
                         ▼               ▼
                    Local Model     Cloud Model
                         │               │
                         └───────┬───────┘
                                 │
                                 ▼
                           AI Response
```

---

# Implemented Architecture

## Phase 1 - Foundation

The foundation establishes the core data and storage layer required by the educational assistant.

Implemented functionality includes:

- Database and storage foundation
- Learner profiles
- Learning sessions
- Interaction records
- Runtime configurations
- Edge-first configurable storage
- Learner/session ownership relationships

The Phase 1 data model provides the foundation used by subsequent phases.

---

## Phase 2 - Basic Text-Based Educational Assistant

Phase 2 introduced the first working AI interaction pipeline.

```text
LearnerProfile
      ↓
LearningSession
      ↓
Text Question
      ↓
Assistant Service
      ↓
LLM Provider
      ↓
Hugging Face Inference API
      ↓
Qwen/Qwen2.5-1.5B-Instruct
      ↓
Generated Response
      ↓
Interaction Persistence
      ↓
Response to Learner
```

Implemented functionality:

- Text-based question API
- Assistant interaction service
- Minimal LLM provider interface
- Hugging Face Inference API integration
- `Qwen/Qwen2.5-1.5B-Instruct`
- Interaction persistence
- Question and response storage
- Latency recording
- Model metadata recording
- Provider error handling
- Environment-based API configuration

Hugging Face-specific functionality is isolated behind the LLM provider/service abstraction so that the implementation can later be replaced by another inference provider.

---

## Phase 3 - Learner Personalization

Phase 3 introduces structured educational personalization.

Implemented functionality includes:

- Preferred language
- Learner competency/proficiency information
- Educational interests/topics
- Learning preferences
- Pre-assessment or competency information
- Personalization context generation
- Structured typed personalization context
- Integration of personalization context with the Assistant Service

The intended flow is:

```text
Learner
   ↓
Learner Profile
   ↓
Educational Preferences
   ↓
Competency Information
   ↓
Personalization Context
   ↓
Assistant
```

Personalization is intentionally limited to educational context and does not implement adaptive policy selection.

---

## Phase 4 - Personalized YouTube Educational Recommendations

Phase 4 introduces educational resource discovery.

```text
Learner Topic / Query
        +
Personalization Context
        ↓
Resource Recommendation Service
        ↓
YouTube Data API
        ↓
Candidate Educational Videos
        ↓
Filtering / Ranking
        ↓
Personalized Recommendations
        ↓
Resource Selection
        ↓
Persistence
```

Implemented functionality includes:

- Topic-based educational resource search
- YouTube Data API integration
- Language-aware recommendations
- Personalization-aware filtering/ranking
- Educational resource metadata handling
- Resource selection
- Learning-session/resource association
- Provider isolation
- Environment-based YouTube API configuration

The system does not scrape YouTube pages.

---

## Phase 5 - Knowledge Preparation + RAG

Phase 5 introduces knowledge preparation and retrieval-augmented generation.

```text
Educational Video
       ↓
Transcript
       ↓
Preprocessing
       ↓
Content Chunks
       ↓
Embeddings
       ↓
FAISS Vector Index
       ↓
Relevant Chunks
       ↓
Retrieved Context
       ↓
LLM
       ↓
Grounded Response
```

Implemented functionality includes:

- Transcript acquisition
- Transcript preprocessing
- Educational content chunking
- Embedding generation
- Embedding storage
- FAISS vector search
- Relevant-context retrieval
- Retrieval-augmented generation
- Passing retrieved context to the LLM

The RAG pipeline is designed around educational content obtained from selected learning resources.

---

## Phase 6 - CRAG / Retrieval Improvement

Phase 6 improves the reliability of the retrieval pipeline.

```text
User Question
     ↓
Initial Retrieval
     ↓
Retrieved Context
     ↓
Retrieval Evaluation
     ↓
Sufficient?
   /      \
 Yes       No
  ↓        ↓
LLM    Refine / Correct Retrieval
            ↓
       Improved Context
            ↓
            LLM
```

Implemented functionality focuses on evaluating retrieved information and improving retrieval when the initial context is insufficient.

The goal is to improve grounding and retrieval quality without introducing the adaptive intelligence planned for later phases.

---

## Phase 7 - Speech-to-Text

Phase 7 introduces speech input.

```text
Student Speech
      ↓
Speech-to-Text
      ↓
Text Question
      ↓
Existing Assistant Pipeline
```

The STT model used is:

```text
ai4bharat/indic-conformer-600m-multilingual
```

Inference is performed through the Hugging Face inference layer rather than local model hosting.

This allows the speech interface to feed into the existing text-based educational assistant without redesigning the core interaction workflow.

---

## Phase 8 - Text-to-Speech

Phase 8 introduces spoken AI responses.

```text
AI Response
     ↓
Text-to-Speech
     ↓
Student Audio
```

The TTS model used is:

```text
ai4bharat/indic-parler-tts
```

Inference is performed through the Hugging Face inference layer.

This completes the initial multimodal interaction loop:

```text
Student
   ↓
Speech / Text
   ↓
Assistant
   ↓
RAG / CRAG
   ↓
AI Response
   ↓
Text / Speech
   ↓
Student
```

---

# Future Phases

The following phases are part of the project roadmap but are **not included in the current Phase 1–8 implementation**.

## Phase 9 - Monitoring + QoE Evaluation

The next planned phase introduces system monitoring and Quality of Experience evaluation.

Metrics will include:

### AI Metrics

- AI inference latency
- Response characteristics
- Model/inference metadata

### Network Metrics

- Network latency
- Connectivity-related measurements
- Relevant network conditions

### System Metrics

- CPU usage
- Memory usage
- Resource utilization

### Interaction Metrics

- Interaction frequency
- Session information
- Response/interaction outcomes

### Learner Feedback

- Explicit learner feedback
- User experience signals

These measurements will be combined into a unified QoE framework:

```text
QoE =
    Learner Experience
    + AI Quality
    + Network Quality
    + UX Quality
```

---

## Phase 10 - Experience Memory

Experience memory will store historical system experiences in the form:

```text
State
  +
Action
  +
Configuration
  +
QoE Outcome
  +
Reward
```

FAISS may later be used to retrieve similar historical states, while complete experience records will remain persisted in the database.

---

## Phase 11 - Adaptive Intelligence

The adaptive intelligence layer will use historical experiences to select appropriate runtime actions.

```text
Current Runtime State
        ↓
Retrieve Similar Experiences
        ↓
Compare Historical Outcomes
        ↓
Select Best Policy
        ↓
Choose Action
        ↓
Modify Predefined Runtime Parameters
```

The adaptive layer will operate on predefined runtime parameters rather than modifying the system architecture dynamically.

---

## Phase 12 - Hybrid Local-Cloud Routing

The system will eventually support genuine local/cloud model routing.

```text
             Query
               │
               ▼
             Router
          ┌────┴────┐
          ▼         ▼
     Local Model  Cloud Model
       Edge       Hugging Face
          └────┬────┘
               ▼
          AI Response
```

Potential routing factors include:

- Query complexity
- Retrieval quality
- Latency
- Network conditions
- Resource availability
- QoE

---

## Phase 13 - Local Model Hosting

After validating the system with Hugging Face inference, selected models may be hosted locally.

This phase will involve:

- Downloading selected small models
- Local/edge inference
- Replacing selected cloud inference calls
- Measuring local inference latency
- Measuring resource utilization
- Comparing local and cloud execution

This is intentionally deferred until the earlier phases are validated.

---

# Technology Direction

The project is designed around a modular architecture so individual components can be replaced without rewriting the complete system.

Key technologies and concepts include:

- Python
- REST API
- Database-backed persistence
- Hugging Face Inference API
- Qwen/Qwen2.5-1.5B-Instruct
- YouTube Data API
- Transcript processing
- Text embeddings
- FAISS
- RAG
- CRAG
- AI4Bharat Indic Conformer
- AI4Bharat Indic Parler-TTS
- Environment-based configuration

---

# Configuration

Configuration is loaded from environment variables and optional `.env` values.

API credentials and other secrets must **never be hard-coded**.

Create a local `.env` file based on `.env.example` and provide the required credentials.

Example configuration values include:

```text
HUGGINGFACE_API_TOKEN=
HUGGINGFACE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
YOUTUBE_API_KEY=
```

The exact variables required by the current implementation are documented in `.env.example`.

---

# Local Setup

Install the project and test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the test suite:

```bash
pytest
```

---

# Testing

The project uses automated tests to validate individual phases while preserving previously implemented functionality.

Testing covers areas such as:

- Database and model behavior
- Learner/session relationships
- API validation
- Assistant interactions
- LLM provider behavior
- Interaction persistence
- Personalization
- Resource recommendation
- YouTube provider behavior
- Knowledge preparation
- Retrieval
- CRAG behavior
- Speech-to-text
- Text-to-speech
- Error handling
- Ownership and data isolation

The complete test suite should be run after implementing each phase:

```bash
pytest
```

Existing tests should continue to pass as new functionality is added.

---

# Architecture Principles

The implementation follows these principles:

### Modular Services

Business logic is separated into appropriate service layers rather than being placed directly in API routes.

### Provider Isolation

External AI and platform integrations are isolated behind provider interfaces where appropriate.

Examples include:

- Hugging Face inference
- YouTube Data API
- STT
- TTS

This allows implementations to be replaced without changing the core application workflow.

### Incremental Development

Each phase introduces only the functionality required for that stage of the roadmap.

Future components are not implemented prematurely.

### Data Ownership and Isolation

Learner data is accessed through the existing learner and learning-session relationships.

One learner must not be able to access another learner's educational or interaction data.

### Configuration Through Environment Variables

Credentials and environment-specific configuration are externalized.

Secrets are never committed to source control.

### Extensibility

The architecture is designed so that later phases can extend the existing system rather than requiring a complete rewrite.

---

# Deferred Functionality

The following functionality is intentionally deferred to later phases:

- Monitoring dashboards
- QoE calculation and evaluation
- Experience memory
- Reward calculation
- Adaptive intelligence
- Policy selection
- Hybrid local/cloud routing
- Local model hosting
- Advanced runtime adaptation

These components should only be introduced in their corresponding roadmap phases.

---

# Documentation

The repository contains additional architecture and implementation documentation.

### `ARCHITECTURE.md`

Contains the intended system architecture, component boundaries, data flow, and phase roadmap.

### `IMPLEMENTATION_STATUS.md`

Tracks the actual implementation status of each project phase, including completed functionality and deferred work.

Both documents should be kept synchronized with the implementation as new phases are completed.

---

# Development Roadmap

```text
Phase 1
Foundation
    ↓
Phase 2
Basic Text Assistant
    ↓
Phase 3
Personalization
    ↓
Phase 4
YouTube Recommendations
    ↓
Phase 5
Knowledge Preparation + RAG
    ↓
Phase 6
CRAG
    ↓
Phase 7
STT
    ↓
Phase 8
TTS
    ↓
Phase 9
Monitoring + QoE
    ↓
Phase 10
Experience Memory
    ↓
Phase 11
Adaptive Intelligence
    ↓
Phase 12
Hybrid Local-Cloud Routing
    ↓
Phase 13
Local Model Hosting
```

---

# Project Vision

The final system is intended to evolve into a **QoE-aware self-adaptive hybrid edge-cloud educational assistant** capable of:

1. Understanding learner input through text and speech.
2. Personalizing educational responses.
3. Recommending relevant educational resources.
4. Building a knowledge base from educational content.
5. Providing grounded answers through RAG and improved retrieval.
6. Supporting multilingual speech interaction.
7. Measuring AI, network, system, and learner experience.
8. Learning from historical system experiences.
9. Selecting better runtime actions based on observed outcomes.
10. Eventually routing workloads between local edge models and cloud models based on runtime conditions.

The implementation follows the principle:

> **Build the basic pipeline first → add personalization and educational content → build RAG → add voice → measure QoE → store experiences → make the system adaptive → introduce genuine local/cloud routing → finally enable local model hosting.**
