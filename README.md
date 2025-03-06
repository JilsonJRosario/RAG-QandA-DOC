# RAG Document Q&A with Groq and Llama

## Overview
This is a Streamlit-based application that utilizes Retrieval-Augmented Generation (RAG) to answer user queries based on research papers. The application employs Groq's `mixtral-8x7b-32768` model along with FAISS for vector-based document retrieval and Ollama embeddings for efficient text representation.

## Features
- Loads and processes research papers in PDF format.
- Splits documents into chunks for efficient embedding.
- Utilizes FAISS for vector-based document retrieval.
- Integrates with Groq's LLM (`mixtral-8x7b-32768`) for generating responses.
- Displays response time for query processing.
- Provides document similarity search for contextual references.

## Installation

### Prerequisites
- Python 3.8+
- Pip
- Groq API Key (stored in `.env` file)

### Setup
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_directory>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the project directory and add the following:
     ```
     GROQ_API_KEY1=your_groq_api_key
     ```
5. Run the Streamlit app:
   ```bash
   streamlit run RAG.py
   ```

## Usage
1. Upload research papers into the `Research_papers` directory.
2. Click the **Document Embedding** button to generate vector embeddings.
3. Enter your query in the text input field.
4. The system will retrieve relevant documents and generate an answer using Groq's LLM.
5. The response time and similarity search results will be displayed.

## File Structure
```
.
├── Research_papers/      # Directory for storing research papers (PDFs)
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not included in repo)
```

## Dependencies
- `streamlit`
- `langchain`
- `langchain_groq`
- `langchain_openai`
- `langchain_community`
- `faiss-cpu`
- `python-dotenv`
