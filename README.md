# Retrieval-Augmented Generation (RAG) Model with SQuAD Dataset

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system utilizing the **SQuAD dataset** for retrieval and **OpenAI GPT** for response generation. The RAG model integrates efficient document retrieval via **Pinecone** and embedding generation using **SentenceTransformer**.

## Features
- **Dynamic Query Embedding**: Generates compact embeddings for queries and documents using `all-MiniLM-L6-v2` from SentenceTransformer.
- **Efficient Retrieval**: Utilizes Pinecone for high-speed vector search with cosine similarity.
- **Generative Response**: Employs OpenAI GPT to generate contextual and relevant answers from retrieved documents.
- **SQuAD Dataset Integration**: Processes the SQuAD dataset to create structured documents with Q&A pairs for fine-tuning retrieval.
- **Rate-Limit Handling**: Implements retry logic with exponential backoff for OpenAI API calls.

## Dataset
- The **Stanford Question Answering Dataset (SQuAD v2.0)** is used to prepare structured documents containing contexts and Q&A pairs.
- Each document includes:
  - **Context**: Paragraphs extracted from SQuAD articles.
  - **Questions**: Relevant questions posed for each context.
  - **Answers**: The corresponding answers for the questions.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/BandaAkshith/Retrieval-Augmented-Generation-RAG-Model-with-SQuAD-Dataset.git]
   cd Retrieval-Augmented-Generation-RAG-Model-with-SQuAD-Dataset
   ```

2. Install the required libraries:
   ```bash
   pip install openai pinecone-client sentence-transformers nltk transformers
   ```

3. Set your API keys securely (Colab recommended):
   - **Pinecone API Key**
   - **OpenAI API Key**

   ```python
   from getpass import getpass
   import os

   os.environ["PINECONE_API_KEY"] = getpass("Enter Pinecone API Key:")
   os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API Key:")
   ```

## Usage

### 1. Dataset Preparation
- Load and parse the SQuAD dataset:
  ```python
  with open('train-v2.0.json', 'r') as file:
      squad_data = json.load(file)
  ```
- Generate embeddings for each document context using SentenceTransformer:
  ```python
  model = SentenceTransformer('all-MiniLM-L6-v2')
  for doc in documents:
      doc["embedding"] = model.encode(doc["text"]).tolist()
  ```

### 2. Pinecone Indexing
- Create or connect to a Pinecone index:
  ```python
  pinecone.create_index(
      name='rag-qa-bot0',
      dimension=384,
      metric='cosine'
  )
  index = pinecone.Index('rag-qa-bot0')
  ```
- Upsert documents with embeddings:
  ```python
  index.upsert([(doc["id"], doc["embedding"], metadata)])
  ```

### 3. Query and Response Generation
- Retrieve documents based on query embeddings:
  ```python
  search_results = index.query(
        vector=query_embedding,
        top_k=1,  # Limit to top 1 for fewer tokens
        include_metadata=True
    )
  ```
- Generate responses using OpenAI GPT:
  ```python
  response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100  # Limit token usage
            )
  ```

## Future Enhancements
1. **Dynamic Query Expansion**: Extract contextual keywords to enhance document retrieval.
2. **Adaptive Fine-Tuning**: Fine-tune generative models with domain-specific datasets for improved responses.
3. **Batch Query Handling**: Add support for batch processing of multiple queries.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments
- [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/)
- [Pinecone](https://www.pinecone.io/)
- [OpenAI](https://openai.com/)
- [Hugging Face](https://huggingface.co/)
