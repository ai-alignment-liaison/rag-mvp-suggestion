# RAG MVP Project

This project implements a Retrieval Augmented Generation (RAG) pipeline using LangGraph, ChromaDB, and OpenAI to build a conversational AI that can answer questions based on a provided knowledge base and generate a Responsible-AI strategy.

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
    *   Using `conda` (preferred due to better dependencies handling):
        ```bash
        conda create -n rag_env python=3.10
        conda activate rag_env
        ```

3.  **Set up your OpenAI API Key:**
    Create a file named `.env` in the root of the project directory and add your OpenAI API key to it:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
4.  **Add link to MIT Risk repository to `.env` file**
    ```env
    MIT_RISKS_SHEET_URL="https://docs.google.com/spreadsheets/d/1f3zgCMTUeqmJ2w2LyiXGo5-UzmrfswxJygbrqEvdOSs/edit?usp=sharing"
    ```

5.  **Install dependencies:**
    Make sure your virtual environment is activated, then run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

The project consists of two main Python scripts: `ingestion_pipeline.py` (optional, for creating/updating the vector database) and `rag_graph.py` (for running the conversational AI).

### 1. (Optional) Data Ingestion

If you want to create or update the vector database with your own documents, run the `ingestion_pipeline.py` script. This script will:
*   Process documents (PDFs and data from a Google Sheet specified in the script).
*   Create embeddings for the documents.
*   Store these embeddings in a ChromaDB vector store located in the `vector_store` directory.

To run the ingestion pipeline:
```bash
python ingestion_pipeline.py
```
If you skip this step, the `rag_graph.py` script will attempt to use an existing `vector_store` directory if present, or might fail if it's not found and required for its operation. The current `rag_graph.py` expects vector stores named "mit_ai_risks" and "responsible_ai_papers".

### 2. Run the RAG Conversational AI

Once the dependencies are installed and the vector store is ready (either by running the ingestion pipeline or using a pre-existing one), you can start the conversational AI by running `rag_graph.py`:

```bash
python rag_graph.py
```

This will start an interactive command-line interface where the AI will first ask you a series of questions to build a user profile. After the interview phase, it will use this profile along with the information in the vector database to generate a Responsible-AI strategy.

## Project Structure

*   `rag_graph.py`: Main script for the conversational AI using LangGraph.
*   `ingestion_pipeline.py`: Script for ingesting and processing documents into ChromaDB.
*   `requirements.txt`: Lists all Python dependencies.
*   `vector_store/`: Directory where ChromaDB stores its data (created by `ingestion_pipeline.py`).
*   `.env`: (You need to create this) For storing API keys and other environment variables. 