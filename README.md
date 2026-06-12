# An ACLED-Based Knowledge Graph for LLM-Driven Conflict Analysis via GraphRAG

Knowledge Graph + GraphRAG pipeline for geopolitical conflict analysis using ACLED data, SPARQL querying, community detection, and LLM-powered question answering.

## Overview

This project investigates how representing armed conflict data as a **Knowledge Graph (KG)** changes what can be queried, analyzed, and reasoned about compared to traditional tabular datasets.

Using conflict data from the **Armed Conflict Location & Event Data Project (ACLED)**, we:

- Build an OWL/RDF Knowledge Graph
- Store and query data through GraphDB
- Detect conflict communities using the Leiden algorithm
- Generate GraphRAG community reports
- Support hybrid question answering via:
  - **Text-to-SPARQL QA**
  - **GraphRAG-based Narrative QA**
- Route user questions automatically through an LLM-based router

---

## Research Questions

1. To what extent does a Knowledge Graph representation expose geopolitical relationships hidden in flat tabular data?

2. Can SPARQL support structured and reproducible conflict analysis?

3. Does the topological structure of the graph reflect real-world conflict dynamics sufficiently well to support LLM reasoning?

---

## Dataset

**Source:** ACLED (Armed Conflict Location & Event Data Project)

- Region: Middle East
- Countries: 15
- Period: 2015–2023
- Events: 91,893

Selected attributes include:

- Event ID
- Event Date
- Event Type
- Actor 1 / Actor 2
- Country
- Coordinates
- Fatalities
- Sources
- Narrative Notes

---

## Architecture

```text
CSV Dataset
      │
      ▼
Knowledge Graph (OWL/RDF)
      │
      ├────────────► GraphDB + SPARQL
      │                    │
      │                    ▼
      │             Text-to-SPARQL QA
      │
      ▼
Leiden Community Detection
      │
      ▼
Community Reports
      │
      ▼
GraphRAG Retrieval
      │
      ▼
Answer Synthesis
      │
      ▼
LLM Router
      │
      ▼
Final Answer
```

---

## Knowledge Graph

### Main Classes

- ConflictEvent
- Actor
- Country
- Source

### Actor Subclasses

- StateForces
- RebelGroup
- PoliticalMilitia
- IdentityMilitia
- Rioters
- Protesters
- Civilian
- ExternalOther

### Object Properties

- hasActor1
- hasActor2
- locatedIn
- reportedBy

---

## Community Detection

The actor-event network is extracted from the Knowledge Graph and analyzed using:

- NetworkX
- Leiden Algorithm

Quality metrics:

| Metric | Value |
|----------|----------|
| Modularity | 0.7202 |
| Average Conductance | 0.0134 |

The resulting communities naturally recover major conflict theatres such as:

- Syria
- Iraq
- Palestine
- Yemen
- Turkey

---

## Hybrid Question Answering

### Text-to-SPARQL

Used for:

- Counts
- Aggregations
- Rankings
- Temporal filters
- Exact answers

Example:

```text
How many times was Israel involved in conflict events?
```

↓

```sparql
SELECT ...
```

↓

```text
Israel was involved in 32,757 events.
```

---

### GraphRAG

Used for:

- Conflict evolution
- Actor dynamics
- Geopolitical summaries
- Cross-community reasoning

Example:

```text
Tell me about Palestine.
```

↓

Narrative answer synthesized from multiple conflict communities.

---

## Evaluation

### SPARQL QA

| Metric | Score |
|----------|----------|
| Router Accuracy | 100% |
| Valid SPARQL Generation | 100% |
| Exact Match Accuracy | 95% |

### GraphRAG

Evaluated using an LLM-as-a-Judge framework inspired by RAGAS.

| Metric | Score |
|----------|----------|
| Faithfulness | 9.3 / 10 |
| Answer Relevance | 9.1 / 10 |

---

## Technology Stack

### Semantic Web

- OWL
- RDF
- Turtle
- SPARQL
- GraphDB
- Protégé

### Python

- Pandas
- NetworkX
- Requests
- RDFLib

### LLMs

- Llama 3.3 70B
- Llama 3.1 8B
- Gemini 2.5 Flash
- Gemini 2.5 Flash Lite

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aronetommaso/Data-Semantics.git
cd Data-Semantics
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Linux / macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Generate the Knowledge Graph

```bash
python preprocessing.py
python rdf_materialization.py
```

### 2. Import into GraphDB

Load:

- ontology.ttl
- acled_kg.ttl

into a GraphDB repository.

### 3. Generate Communities

```bash
python community_detection.py
```

### 4. Generate Community Reports

```bash
python report_generation.py
```

### 5. Run the QA System

```bash
python graph_rag.py
```

or

```bash
python text_to_sparql.py
```

---

## Limitations

- ACLED coverage limited to the Middle East subset.
- Community reports may lose fine-grained temporal information.
- LLM routing can occasionally misclassify temporal aggregation queries.
- GraphRAG evaluation relies on LLM-as-a-Judge rather than domain experts.

---

## Authors

- **Tommaso Arone**
- **Clelia Meloni**
- **Lorenzo Triolo**

Data Semantics — University of Milano-Bicocca

---

## License

This repository is released for academic and research purposes.
