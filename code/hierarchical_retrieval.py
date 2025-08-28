"""
Document-level retrieval from the IPOSGPT database.

This function demonstrates the first stage of retrieving documents 
based on metadata filters and similarity search using pgvector.

Note: This code is provided as a building block and requires 
a PostgreSQL database with pgvector installed. 
You must configure your own database connection via environment variables 
or a config file before running.
"""

import os
import psycopg2


def document_level_retrieval(
    query_text: str,
    recency_option: str = "All Dates",
    year: int = None,
    start_date: str = None,
    end_date: str = None,
    knowledge_categories: list = None,
    open_access_types: list = None,
    data_sources: list = None,
    source_types: list = None,
    is_peer_reviewed: bool = None,
    top_k: int = 20,
):
    """
    Retrieve documents from the database with filtering and semantic similarity.

    Args:
        query_text (str): Search query for semantic retrieval.
        recency_option (str): One of ["All Dates", "In this year", "Since this year", "Between these dates"].
        year (int, optional): Year filter.
        start_date (str, optional): Start date (YYYY-MM-DD).
        end_date (str, optional): End date (YYYY-MM-DD).
        knowledge_categories (list, optional): Filter by knowledge categories.
        open_access_types (list, optional): Filter by access type.
        data_sources (list, optional): Filter by data source.
        source_types (list, optional): Filter by source type.
        is_peer_reviewed (bool, optional): Filter by peer review status.
        top_k (int, optional): Number of top results to retrieve.

    Returns:
        list or None: Retrieved records as a list of tuples, or None if no results.
    """

    # --- Database credentials should come from environment variables ---
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_NAME = os.getenv("DB_NAME", "database")
    DB_PORT = os.getenv("DB_PORT", "xxxx")

    try:
        # Establish a connection to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            port=DB_PORT,
        )
        cursor = connection.cursor()

        # Base SQL query
        sql_query = """
        SELECT 
            CASE 
                WHEN data_source = 'OpenAlex' AND doi IS NOT NULL 
                     AND doi NOT IN ('NA', 'nan', 'NaN') 
                THEN doi
                ELSE href
            END AS link_identifier,
            title, 
            body, 
            publication_date, 
            journal,
            author_list,
            data_source,
            source_type,
            is_peer_reviewed,
            source_category
        FROM documents
        WHERE 1=1
        """

        params = []

        # Date filters
        if recency_option == "In this year" and year:
            sql_query += " AND EXTRACT(YEAR FROM publication_date) = %s"
            params.append(int(year))
        elif recency_option == "Between these dates" and start_date and end_date:
            sql_query += " AND publication_date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        elif recency_option == "Since this year" and year:
            sql_query += " AND publication_date >= %s"
            params.append(f"{year}-01-01")

        # Category filters
        if knowledge_categories and "All" not in knowledge_categories:
            sql_query += " AND source_category IN %s"
            params.append(tuple(knowledge_categories))

        if data_sources and "All" not in data_sources:
            sql_query += " AND data_source IN %s"
            params.append(tuple(data_sources))

        if source_types and "All" not in source_types:
            sql_query += " AND source_type IN %s"
            params.append(tuple(source_types))

        if is_peer_reviewed and "All" not in is_peer_reviewed:
            sql_query += " AND is_peer_reviewed IN %s"
            params.append(tuple(is_peer_reviewed))

        if open_access_types:
            sql_query += " AND has_data IN %s"
            params.append(tuple(open_access_types))

        # Similarity search with pgvector
        sql_query += """
        ORDER BY document_embedding <-> embedding('text-multilingual-embedding-002', %s)::vector
        LIMIT %s;
        """
        params.extend([query_text, top_k])

        cursor.execute(sql_query, params)
        results = cursor.fetchall()

        cursor.close()
        connection.close()

        return results if results else None

    except Exception as e:
        print(f"[Error] Document retrieval failed: {e}")
        return None


#################################################################################################

"""
Process documents by splitting them into chunks, embedding them, 
and ranking them by similarity to the query.

This function is provided as a building block and not as 
production-ready code. It demonstrates chunking, batch processing, 
and cosine similarity scoring.
"""

import numpy as np
import tiktoken
from vertexai.language_models import TextEmbeddingModel


def process_documents_with_chunking(
    query_text: str,
    documents: list,
    chunk_size: int = 1500,
    top_chunks: int = 50,
    max_tokens_per_batch: int = 15000,
):
    """
    Chunk documents, embed them, and rank by similarity to a query.

    Args:
        query_text (str): The query string.
        documents (list): List of documents, where each document is a tuple:
            (link_identifier, title, body, publication_date, journal, authors,
             data_source, source_type, is_peer_reviewed, source_category).
        chunk_size (int): Character length of each chunk.
        top_chunks (int): Number of top chunks to return.
        max_tokens_per_batch (int): Maximum token budget for embedding requests.

    Returns:
        list: Ranked chunks (dicts) with similarity scores and metadata.
    """

    # --- Load embedding model and tokenizer ---
    embedding_model = TextEmbeddingModel.from_pretrained(
        "text-multilingual-embedding-002"
    )
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # --- Compute query embedding ---
    query_embedding = embedding_model.get_embeddings([query_text])[0].values
    ranked_chunks = []

    for doc in documents:
        (
            link_identifier,
            title,
            body,
            publication_date,
            journal,
            authors,
            data_source,
            source_type,
            is_peer_reviewed,
            source_category,
        ) = doc

        # --- Step 1: Split document into chunks ---
        chunks = [body[i: i + chunk_size] for i in range(0, len(body), chunk_size)]
        chunk_token_counts = [len(tokenizer.encode(chunk)) for chunk in chunks]

        # --- Step 2: Process chunks in token-limited batches ---
        current_batch, current_batch_tokens, batch_results = [], 0, []
        for i, chunk in enumerate(chunks):
            tokens = chunk_token_counts[i]

            # Skip overly large chunks
            if tokens > max_tokens_per_batch:
                print(f"[Warning] Skipping chunk {i} (> {max_tokens_per_batch} tokens).")
                continue

            # Process current batch before exceeding token budget
            if current_batch_tokens + tokens > max_tokens_per_batch:
                batch_results.extend(embedding_model.get_embeddings(current_batch))
                current_batch, current_batch_tokens = [], 0

            # Add chunk to current batch
            current_batch.append(chunk)
            current_batch_tokens += tokens

        # Process remaining chunks
        if current_batch:
            batch_results.extend(embedding_model.get_embeddings(current_batch))

        # --- Step 3: Compute similarity scores ---
        for i, chunk in enumerate(chunks[:len(batch_results)]):
            chunk_vector = batch_results[i].values
            similarity_score = np.dot(query_embedding, chunk_vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_vector)
            )
            ranked_chunks.append(
                {
                    "chunk": chunk,
                    "similarity_score": float(similarity_score),
                    "link_identifier": link_identifier,
                    "title": title,
                    "publication_date": publication_date,
                    "journal": journal,
                    "authors": authors,
                    "data_source": data_source,
                    "source_type": source_type,
                    "is_peer_reviewed": is_peer_reviewed,
                    "source_category": source_category,
                }
            )

    # --- Step 4: Rank and return top chunks ---
    ranked_chunks = sorted(ranked_chunks, key=lambda x: x["similarity_score"], reverse=True)[
        :top_chunks
    ]

    return ranked_chunks
