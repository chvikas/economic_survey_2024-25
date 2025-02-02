import os
import time
import json
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

import hashlib

load_dotenv()

# Use HuggingFaceEmbeddings to wrap SentenceTransformer
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

pinecone_api_key = os.getenv("PINECONE_API_KEY")

def generate_document_id(doc):
    """Generates a unique ID for a document based on its content."""
    content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
    return f"doc_{content_hash}"

def save_progress(processed_ids: List[str], progress_file: str = 'indexing_progress.json'):
    """Save the IDs of processed documents to a file."""
    with open(progress_file, 'w') as f:
        json.dump({"processed_ids": processed_ids}, f)

def load_progress(progress_file: str = 'indexing_progress.json') -> List[str]:
    """Load the IDs of previously processed documents."""
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return data.get("processed_ids", [])
    except FileNotFoundError:
        return []

def setup_pinecone_vectorstore(documents, index_name="economic-survey-2025",
                             namespace="economic-survey-2025",
                             batch_size=100,
                             use_local_embeddings=True):
    """Sets up the Pinecone index and processes documents efficiently with progress tracking."""

    pc = Pinecone(api_key=pinecone_api_key)

    if use_local_embeddings:
        embedding_dimension = len(embeddings.embed_query("test"))
    else:
        raise Exception("Pinecone embeddings disabled - switch use_local_embeddings=True")

    # Pinecone setup
    cloud = os.getenv('PINECONE_CLOUD') or 'aws'
    region = os.getenv('PINECONE_REGION') or 'us-east-1'
    spec = ServerlessSpec(cloud=cloud, region=region)

    # Create index if it doesn't exist
    indexes = pc.list_indexes()
    if index_name not in [index.name for index in indexes]:
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",
            spec=spec
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pc.Index(index_name)
    docsearch = PineconeVectorStore(index, embeddings, namespace=namespace)

    # Load previous progress
    processed_ids = load_progress()
    print(f"Found {len(processed_ids)} previously processed documents")

    # Generate IDs for all documents upfront
    all_doc_ids = [generate_document_id(doc) for doc in documents]
    remaining_doc_ids = set(all_doc_ids) - set(processed_ids)
    remaining_docs = [doc for doc, doc_id in zip(documents, all_doc_ids) if doc_id in remaining_doc_ids]

    total_documents = len(documents)
    num_processed = len(processed_ids)

    print(f"Total documents to process: {total_documents}")
    print(f"Previously processed documents: {num_processed}")
    print(f"Remaining documents to process: {len(remaining_docs)}")

    if not remaining_docs:
        print("No new documents to index. Skipping indexing process.")
        return docsearch  # Exit early if no new documents

    print(f"Processing {len(remaining_docs)} remaining documents in batches of {batch_size}")

    # Process documents in batches
    for i in tqdm(range(0, len(remaining_docs), batch_size)):
        batch = remaining_docs[i:i + batch_size]

        # Prepare batch data
        texts_to_index = []
        metadatas_to_index = []
        ids_to_index = []
        batch_processed_ids = []

        for doc in batch:
            doc_id = generate_document_id(doc) # Generate ID here for remaining docs
            texts_to_index.append(doc.page_content)
            metadatas_to_index.append(doc.metadata)
            ids_to_index.append(doc_id)
            batch_processed_ids.append(doc_id)

        if texts_to_index:
            try:
                docsearch.add_texts(texts_to_index, metadatas=metadatas_to_index, ids=ids_to_index)
                processed_ids.extend(batch_processed_ids)
                save_progress(processed_ids)
                # Add a small delay to avoid rate limits
                time.sleep(0.1)
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Save progress even if there's an error
                save_progress(processed_ids)
                raise

    print(f"Successfully processed all new documents. Total documents indexed: {len(processed_ids)}")
    return docsearch

if __name__ == '__main__':
    from data_loader import load_and_split_data

    file_path = '../chunks.json'
    splits = load_and_split_data(file_path)

    # Setup Pinecone with local embeddings and batch processing
    docsearch = setup_pinecone_vectorstore(splits, batch_size=100, use_local_embeddings=True)
    print("Pinecone vector store setup and document indexing complete.")