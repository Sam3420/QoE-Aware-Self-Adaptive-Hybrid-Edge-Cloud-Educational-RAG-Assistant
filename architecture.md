# QoE-Aware Self-Adaptive Hybrid Edge–Cloud Educational RAG Assistant

## 1. Project Overview

This project implements a personalized educational assistant that helps learners discover educational resources, interact with learning content, and receive AI-generated answers grounded in selected educational material.

The system combines personalized educational resource recommendation, Retrieval-Augmented Generation (RAG), hybrid local–cloud inference, Quality of Experience (QoE) monitoring, and experience-based runtime adaptation.

The long-term objective is to create a closed-loop educational system that learns from previous learner-system interactions and adapts future runtime configurations to improve learner QoE while considering resource efficiency.

---

## 2. High-Level Architecture

The system follows this conceptual flow:

Learner
→ Interaction & Personalization
→ Educational Resource Recommendation
→ Knowledge Preparation
→ Hybrid RAG
→ Educational Response
→ Monitoring & QoE Evaluation
→ Experience Memory
→ Adaptive Policy Selection
→ Runtime Adaptation
→ Future Learning Sessions

The implementation should be modular. Components must communicate through clearly defined data models and interfaces.

---

## 3. Core Architectural Principles

1. Build the application as a modular monolith. Do not introduce microservices unless explicitly required.
2. Each module must have clearly defined inputs and outputs.
3. Use typed data models instead of passing unstructured dictionaries between modules.
4. Keep persistent application data separate from vector search indexes.
5. FAISS is used for vector similarity search only. Complete records and metadata are stored in the relational database.
6. All important interactions, configurations, and outcomes must be logged from the beginning.
7. Components should be replaceable through interfaces where practical.
8. Do not implement future modules before their implementation phase.
9. Avoid unnecessary infrastructure and dependencies.
10. Do not rewrite unrelated modules when implementing a new feature.

---

## 4. Major Modules

### 4.1 Learner Interaction

Handles learner authentication, session management, text interaction, and later speech-based interaction.

### 4.2 Personalization

Manages learner profiles, preferences, competency levels, assessment results, learning history, and personalized learning context.

### 4.3 Educational Resource Recommendation

Uses the YouTube Data API to retrieve educational videos based on learner topics and preferences.

Competency level may influence query construction and ranking, but the system must not assume that YouTube provides reliable difficulty labels.

### 4.4 Knowledge Preparation

Processes selected educational resources into a searchable knowledge base.

Pipeline:

Video
→ Transcript
→ Preprocessing
→ Chunking
→ Embeddings
→ FAISS Vector Index

Previously processed videos should be reused instead of being processed repeatedly.

### 4.5 Hybrid RAG

Answers learner questions using retrieved educational context.

The pipeline will eventually include retrieval quality evaluation and hybrid routing between local and cloud language models.

### 4.6 Monitoring and QoE Evaluation

Collects learner, AI, network, and system performance metrics.

Metrics must be stored as raw observations so QoE calculations can be recomputed during experimentation.

### 4.7 Experience Memory

Stores historical experiences produced by learning sessions and runtime adaptations.

An experience conceptually contains:

State + Action/Policy + Outcome + QoE + Resource Efficiency + Reward

FAISS may be used to retrieve similar state vectors, while complete experience records remain in the database.

### 4.8 Adaptive Intelligence

Uses similar historical experiences to select from predefined runtime policies.

The adaptation engine may change only approved runtime configuration parameters. It must never modify source code or generate unrestricted system changes.

---

## 5. Core Data Models

The following entities should remain conceptually stable across the project:

- LearnerProfile
- LearningSession
- Interaction
- EducationalResource
- KnowledgeDocument
- KnowledgeChunk
- RuntimeConfiguration
- RuntimeMetrics
- QoEScore
- RuntimeState
- Experience
- RuntimePolicy

Relationships should be explicitly defined using database models and typed application models.

---

## 6. Data Storage Architecture

### Relational Database

Stores:

- Learner profiles
- Learning sessions
- Interactions
- Educational resource metadata
- Runtime configurations
- Monitoring metrics
- QoE records
- Historical experiences
- Policy history

### FAISS

Stores/searches:

- Educational content embeddings
- Runtime state embeddings for experience similarity search

FAISS should return identifiers that are used to retrieve complete records from the database.

---

## 7. Technology Stack

### Backend and Application

- Python 3.11+
- FastAPI
- Pydantic
- SQLAlchemy

### Database

- SQLite for initial development and experimentation
- PostgreSQL if a production-scale database becomes necessary

### AI and LLM Integration

- PyTorch
- Hugging Face Transformers
- Configurable local Small Language Model (SLM)
- Configurable cloud Large Language Model (LLM)

LLM providers must be abstracted behind interfaces so models can be changed without modifying application logic.

### RAG and Retrieval

- FAISS for vector similarity search
- Sentence Transformers or another configurable embedding model

### Speech Processing

- Whisper / IndicWhisper for Speech-to-Text
- Configurable TTS provider

Speech modules should be implemented as replaceable adapters.

### Educational Resource Recommendation

- YouTube Data API v3

### Monitoring and Observability

- OpenTelemetry for application-level tracing and instrumentation
- Prometheus for system and service metrics where required
- Grafana for metric visualization
- LangSmith for LLM tracing and experimentation where applicable

Monitoring tools should not be tightly coupled to core application logic.

### Evaluation

- RAGAS for RAG evaluation
- DeepEval for LLM response evaluation
- JiWER for Speech-to-Text evaluation
- Pandas and NumPy for educational metrics and experimental analysis

### Development and Testing

- pytest
- Git and GitHub
- Environment-based configuration using `.env`

---

## 8. Implementation Rules

### Build Incrementally

The implementation order is:

1. Project Foundation and Data Layer
2. Basic Text-Based Educational Assistant
3. Learner Profiles and Personalization
4. Educational Resource Recommendation
5. Knowledge Preparation Pipeline
6. RAG-Based Educational Question Answering
7. CRAG and Hybrid Local–Cloud Routing
8. Speech Input and Output
9. Monitoring and QoE Evaluation
10. Experience Memory and Resource Efficiency
11. Adaptive Policy Selection
12. Closed-Loop Runtime Adaptation
13. Experimental Comparison and Validation

Do not skip directly to the adaptive intelligence module.

### Data Collection

Even before QoE and adaptation are implemented, the system should record:

- Session identifiers
- Interaction timestamps
- Selected runtime configuration
- Response latency
- Model/provider information
- Errors and outcomes

This historical data will support later experimentation and adaptive intelligence.

### Configuration

Runtime parameters must be externalized and configurable.

Examples of future configurable parameters include:

- Retrieval Top-K
- Model selection
- Routing preference
- Response length
- Speech rate

The system should use predefined configuration options rather than unrestricted parameter modification.

---

## 9. Testing Requirements

Each module should include:

1. Unit tests for core functionality.
2. Basic failure and edge-case tests.
3. Clear success criteria before integration.

Before modifying existing code, inspect the current implementation and preserve working functionality.

---

## 10. Important Implementation Constraints

- Do not overengineer the system.
- Prefer simple, working implementations over complex infrastructure.
- Do not fabricate metrics or data.
- Do not claim unsupported personalization capabilities from external APIs.
- Preserve raw experimental data for later analysis.
- Keep baseline implementations configurable within the same codebase where possible.
- Ensure different learners and sessions remain isolated from each other.

---

## 11. Development Philosophy

The system should be developed as a sequence of tested vertical capabilities.

Each new module should integrate with existing data models rather than bypassing them.

The objective is not to build every component at once. The objective is to maintain a stable foundation so that monitoring, QoE evaluation, and experience-based adaptation can be added without redesigning the earlier system.