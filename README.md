# RAG MVP Project

This project implements a Retrieval Augmented Generation (RAG) pipeline using LangGraph, ChromaDB, and OpenAI. It features a conversational AI that conducts a sequential, two-stage interview to understand a user's context and then generates a tailored Responsible AI strategy based on the conversation and a knowledge base of relevant documents.

## Prerequisites

*   Python 3.10 or higher
*   An OpenAI API Key

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:ai-alignment-liaison/RAG_MVP_Suggestion.git
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment:**

    *   Using `venv`:
        ```bash
        python3 -m venv rag_env
        source rag_env/bin/activate
        ```
    *   Using `conda`:
        ```bash
        conda create -n rag_env python=3.10
        conda activate rag_env
        ```

3.  **Set up your OpenAI API Key:**
    Create a `.env` file in the project's root directory and add your key:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
4.  **Add the MIT AI Risks Google Sheet URL to the `.env` file:**
    ```env
    MIT_RISKS_SHEET_URL="https://docs.google.com/spreadsheets/d/1f3zgCMTUeqmJ2w2LyiXGo5-UzmrfswxJygbrqEvdOSs/edit?usp=sharing"
    ```

5.  **Install dependencies:**
    With your virtual environment active, run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

The project consists of two main Python scripts: `ingestion_pipeline.py` and `rag_graph.py`.

### 1. Data Ingestion (Optional)

To create or update the vector database with your own documents, run the ingestion pipeline. The script processes PDFs from the `/papers` directory and data from the Google Sheet specified in your `.env` file. It then creates and stores their embeddings in a ChromaDB vector store inside the `vector_store/` directory.

To run the ingestion pipeline:
```bash
python ingestion_pipeline.py
```
If you skip this step, the application will use the existing `vector_store` directory.

### 2. Run the Conversational AI

To start the conversational AI, run the `rag_graph.py` script:

```bash
python rag_graph.py
```

This will launch an interactive command-line interview process. The AI will guide you through two stages:

1.  **Predefined Interview:** The AI will ask five required questions to build a foundational user profile (industry, region, audience, use cases, and risk hypotheses).
2.  **Free-Form Interview:** After the predefined questions, the AI will transition to a conversational interview to gather more nuanced information about your values and understanding of AI. This stage has a few rules:
    *   It will ask a maximum of five questions.
    *   The interview will stop automatically if you use phrases like "stop" or "exit."

After the interview is complete, the AI will use the collected information and the knowledge base to generate and refine a comprehensive Responsible AI strategy for you.

## Project Structure

*   `rag_graph.py`: The main script for the conversational AI, built with LangGraph.
*   `ingestion_pipeline.py`: Ingests and processes documents into the ChromaDB vector store.
*   `requirements.txt`: Lists all required Python dependencies.
*   `vector_store/`: The directory where ChromaDB stores its vector data.
*   `.env`: (You need to create this) For storing API keys and environment variables.
*   `papers/`: A directory containing research papers in PDF format for ingestion. 