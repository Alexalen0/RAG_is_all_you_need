# PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions based on PDF documents using OpenVINO and TinyLlama.

## Features

- PDF text extraction
- Semantic search using FAISS
- Question answering using TinyLlama with OpenVINO optimization
- User-friendly Streamlit interface

## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM
- Intel CPU (for OpenVINO optimization)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Deeplearners3056/RAG_is_all_you_need.git rag
    ```

2. Navigate to the project directory:

    ```bash
    cd rag
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:
``` bash
streamlit run openvino.py
```

## Docker Instructions

### Build Docker Image

To build the Docker image, run the following command in the project directory:

```bash
docker build -t pdf-rag-system .
```

### Run Docker Container

To run the Docker container, use the following command:

```bash
docker run -p 8501:8501 pdf-rag-system
```
