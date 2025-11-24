# 🚀 Apache Jira Scraper + LLM Training Pipeline + Streamlit Dashboard

### **Enterprise-Grade Data Engineering + ML Pipeline**

This repository showcases a **real-world production-grade pipeline**
that extracts data from **Apache's public Jira**, cleans & transforms it
into **LLM‑ready JSONL datasets**, and visualizes insights using a
**beautiful Streamlit dashboard with AI-powered analytics**.

Perfect for demonstrating **Data Engineering + AI/ML + MLOps** skills.

------------------------------------------------------------------------

# 🌟 Key Highlights

### ✔ Real Apache Jira Web Scraping

-   Issues, comments, metadata, timestamps\
-   Pagination, retries, resume state\
-   Handles 429, 5xx, malformed data

### ✔ LLM Data Transformation

-   Clean natural text\
-   Summaries, Q/A, classifications\
-   JSONL format used for LLM training

### ✔ Streamlit + LLM Dashboard

-   Semantic search\
-   Topic clustering\
-   Issue analytics visualizations\
-   Chat with Jira dataset\
-   Embedding visualizer

### LLM-Powered Semantic Search & Analytics (OpenAI-Integrated)

 - Integrates OpenAI GPT models for semantic search, natural-language querying, and intelligent issue understanding.
 - Converts Jira issues into dense vector embeddings for high-precision similarity search and context retrieval.
 - Supports AI-driven summaries, root-cause extraction, pattern detection, and domain-aware insights.
 - Enables users to ask natural questions like “What are the top recurring failures in HDFS?” and get analysis-grade answers.
 - Combines data engineering + vector search + LLM reasoning into a streamlined production-style pipeline.
 - Embedded inside the Streamlit dashboard for real-time, interactive AI analytics over large Jira datasets.

### ✔ Production-Ready Engineering

-   CI/CD pipeline\
-   Complete test suite\
-   Modular & extensible architecture\
-   Docker support

------------------------------------------------------------------------

# 📁 Project Structure

    Jira-Web-Scraping-For-LLM-training/
    ├── .github/
    │   └── workflows/
    │       └── ci.yml
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── samples/
    │       └── HADOOP_sample.jsonl
    ├── docs/
    │   └── architecture.md
    ├── notebooks/
    │   └── quick_demo.py
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── jira_scraper.py
    │   ├── transform.py
    │   ├── utils.py
    │   └── cli.py
    ├── streamlit_app/
    │   ├── dashboard.py
    │   └── llm_utils.py
    ├── tests/
    │   ├── test_transform.py
    │   └── test_scraper.py
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .gitignore
    └── README.md

------------------------------------------------------------------------

# 🧠 Architecture Overview

                 ┌────────────────────────────┐
                 │      Apache Jira API        │
                 └──────────────┬─────────────┘
                                │ (JSON REST)
                   ┌────────────▼──────────────┐
                   │       Scraper Layer       │
                   │ (Rate limits, retries,    │
                   │  pagination, resume state)│
                   └────────────┬──────────────┘
                                │
          ┌─────────────────────▼────────────────────┐
          │               Raw Storage                 │
          │        data/raw/{PROJECT}.jsonl          │
          └─────────────────────┬────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │    Transformation       │
                    │  (clean text, enrich,   │
                    │   LLM tasks, JSONL)     │
                    └───────────┬────────────┘
                                │
                 ┌──────────────▼────────────────────┐
                 │         Processed Storage          │
                 │   data/processed/{PROJECT}.jsonl   │
                 └──────────────┬────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Streamlit Dashboard │
                     │ (LLM Q/A, insights)  │
                     └──────────────────────┘

------------------------------------------------------------------------

# ⚙️ Setup Instructions

## **1️⃣ Clone the repo**

    git clone https://github.com/ManojPrathapa/Jira-Web-Scraping-For-LLM-training.git
    cd Jira-Web-Scraping-For-LLM-training

## **2️⃣ Create virtual environment**

    python -m venv venv
    source venv/bin/activate      # Linux/macOS
    venv\Scripts\activate         # Windows

## **3️⃣ Install dependencies**

    pip install -r requirements.txt

## **4️⃣ Run the scraper**

(default projects: `HADOOP SPARK KAFKA`)

    python -m src.cli --mode scrape

## **5️⃣ Run the transformer**

    python -m src.cli --mode transform

## **6️⃣ Start the Streamlit dashboard**

    streamlit run streamlit_app_fast.py

------------------------------------------------------------------------

# 🧩 Detailed Design Reasoning

## **1. Scraper Layer (Fault-Tolerant + Resume Support)**

### 💥 Handles failure scenarios:

-   API rate limits\
-   HTTP 429 & 5xx responses\
-   Missing fields in Jira response\
-   Empty/malformed issues\
-   Pagination edge cases\
-   Interrupted runs with resume state

### 🧠 Techniques:

-   Stateless & stateful mixed architecture\
-   Retry-on-failure with exponential backoff\
-   Request session pooling\
-   Local checkpointing

------------------------------------------------------------------------

## **2. Transformation Layer**

Ensures **clean, consistent, LLM-ready** text.

### Includes:

-   HTML → Markdown → Plain text cleanup\
-   Issue + comment thread merging\
-   Metadata normalization\
-   Derived datasets:
    -   Summaries\
    -   Classifications\
    -   Q&A pairs\
    -   Topic tags

All exported in **JSONL**, compatible with: - OpenAI fine‑tuning\
- Anthropic Claude\
- HuggingFace LLaMA\
- Google Gemini

------------------------------------------------------------------------

## **3. Streamlit Dashboard**

A polished dashboard with:

### 📊 Analytics:

-   Issue volumes & trends\
-   Status & priority distribution\
-   Label frequency heatmaps\
-   User activity patterns\
-   Word clouds

### 🤖 AI Features:

-   Semantic search\
-   LLM-based Q&A\
-   Chatbot trained on Jira issues\
-   Embedding visualizer (UMAP/TSNE)\
-   Cluster explorer

------------------------------------------------------------------------

# 🐞 Edge Cases Handled

### Scraper

-   Empty "fields" section\
-   Null assignee/reporter\
-   HTML with broken tags\
-   Unicode issues\
-   Comments missing timestamps\
-   API pagination breaks mid-page\
-   Interrupted write → safely recoverable

### Transformer

-   Missing description field\
-   Overlapping HTML entities\
-   Multiline descriptions\
-   Comments with code blocks\
-   Emojis & unicode normalization\
-   Unexpected schema changes

------------------------------------------------------------------------

# 🚀 Optimizations

### Performance:

-   Future-ready concurrency design\
-   Cached HTTP session\
-   Streamed writes to `.jsonl` files\
-   Minimal repeated parsing overhead

### Reliability:

-   Local checkpointing\
-   Graceful crash recovery\
-   Full logging & instrumentation

### LLM Data Quality:

-   Aggressive noise filtering\
-   Deterministic formatting\
-   Clean + consistent JSON schema

### Screen Shots:

<img width="1906" height="1136" alt="Screenshot 2025-11-25 012750" src="https://github.com/user-attachments/assets/1e700c81-5c1b-43c0-88ee-940b29c7efb2" />

<img width="1905" height="1037" alt="Screenshot 2025-11-25 012905" src="https://github.com/user-attachments/assets/ef22c85b-c00c-4b9f-b6e0-a7f5d53ebd12" />

------------------------------------------------------------------------

# 🔮 Future Improvements

-   Multi-threaded or async scraping\
-   Distributed scraping with Ray\
-   Embeddings stored in vector DBs\
-   Pinecone / Weaviate integration\
-   Auto QC scoring of generated tasks\
-   Model fine‑tuning notebook\
-   Add multi-agent summarization pipeline

------------------------------------------------------------------------

# 🙌 Author

Built by Manoj Prathapa, combining Data Engineering, Web Scraping,
ML Pipelines, LLM Architecture, DevOps, and Visualization
Engineering.
------------------------------------------------------------------------
