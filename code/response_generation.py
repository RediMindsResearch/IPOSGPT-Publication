"""
This script demonstrates how retrieved documents are used to generate a response with an LLM. 

Note: 
- You must replace the embedding model, DB retrieval, and LLM backend with your own setup for production use.
"""

import random
from typing import List, Dict
import google.generativeai as genai

# Import functions from your other modules
from hierarchical_retrieval import document_level_retrieval
from hierarchical_retrieval import process_documents_with_chunking


# --- Example Query ---
query_text = "What national-level strategies are most effective in addressing the issue of ghost fishing gear?"

#System Prompt (This version is partial, not the final version used in production)
system_prompt = """
You are a highly specialized AI assistant. You will be presented with a user query, followed by a corpus of source material. Each source will come with a source number and content in the format - **[{**source number**}] \n Content: content**. Your task is to use this source material and ONLY this source material to answer the user query. Do not use any knowledge you may have that is not present in the provided source material.

"""


# --- Document-level retrieval call ---
retrieval_results = document_level_retrieval(
    query_text="What national-level strategies are most effective in addressing the issue of ghost fishing gear?",
    recency_option="All Dates",
    year=None,
    start_date=None,
    end_date=None,
    knowledge_categories=None,
    open_access_types=None,
    data_sources=None,
    source_types=None,
    is_peer_reviewed=None,
    top_k=20)


# --- Chunking and ranking ---
processed_docs = process_documents_with_chunking(
    query_text="What national-level strategies are most effective in addressing the issue of ghost fishing gear?",
    documents=retrieval_results,
    chunk_size=1500,
    top_chunks=50,
    max_tokens_per_batch=15000)


# --- Response generation ---
if processed_docs:
    relevant_texts, sources, org_links = [], [], set()

    for doc in processed_docs:
        url = doc["link_identifier"]
        title = doc["title"]
        publication_date = doc["publication_date"]
        publisher = doc["journal"]
        authors = doc["authors"]
        data_source = doc["data_source"]
        source_type = doc["source_type"]
        is_peer_reviewed = doc["is_peer_reviewed"]
        knowledge_category = doc["source_category"]

        if url not in org_links:
            org_links.add(url)
            sources.append(
                {
                    "title": title,
                    "url": url,
                    "date": publication_date,
                    "knowledge_category": knowledge_category,
                    "publisher": publisher,
                    "authors": authors,
                    "data_source": data_source,
                    "source_type": source_type,
                    "is_peer_reviewed": is_peer_reviewed,
                }
            )

    # Map URLs → source numbers
    org_links = list(org_links)
    url_to_source_number, source_number = {}, 1
    for source in sources:
        url = source["url"]
        if url not in url_to_source_number:
            url_to_source_number[url] = source_number
            source_number += 1

    # Build context text
    for doc in processed_docs:
        url = doc["link_identifier"]
        content = doc["chunk"]
        source_number = url_to_source_number.get(url, "unknown")
        relevant_texts.append(f"[{source_number}] \n Content: {content}")

    context_text = "\n\n".join(relevant_texts)

    # Construct final prompt
    messages = f"""
    {system_prompt}

    Context:
    {context_text}

    User Query:
    {query_text}
    """

    # LLM output
    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=8192
        )
    )

    response = model.generate_content(messages)
    response_text = response.text
    
    print("=== Generated Response ===")
    print(response_text)
else:
    print("No relevant documents found.")
