import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import argparse
from pathlib import Path

# Lists to classify schemas by their index format
cypherbench = ['company', 'fictional_character', 'flight_accident', 'geography', 'movie', 'nba', 'politics', 'soccer', 'terrorist_attack']
mtq = ['bloom50', 'covid', 'er', 'gdsc', 'healthcare', 'legis_graph', 'osm', 'pole', 'twitter_trolls', 'wwc']

# Paths come from config (env-overridable).
from config import EMBED_MODEL_PATH as MODEL_PATH, FAISS_INDEX_DIR as INDEX_DIR

def search(schema_name, query_text, k=10):
    """
    Searches for similar attribute values in the specified schema's FAISS index.
    It uses predefined lists to determine the correct index format and search logic.
    """
    # --- 1. Determine Index Format and Set Paths based on schema_name ---
    index_format = None
    if schema_name in cypherbench:
        index_format = 'cypherbench'
        index_path = INDEX_DIR / schema_name / 'attr.index'
        meta_path = INDEX_DIR / schema_name / 'attr.meta.json'
        print(f"Schema '{schema_name}' is in cypherbench list. Using Cosine Similarity format.")
    elif schema_name in mtq:
        index_format = 'mtq'
        index_path = INDEX_DIR / schema_name / 'attr.index'
        meta_path = INDEX_DIR / schema_name / 'attr.meta.json'
        print(f"Schema '{schema_name}' is in mtq list. Using L2 Distance format.")
    else:
        print(f"Error: Schema '{schema_name}' not found in 'cypherbench' or 'mtq' lists.")
        return

    if not index_path.exists() or not meta_path.exists():
        print(f"Index or metadata file not found at expected path for schema '{schema_name}'.")
        print(f"Checked for: {index_path}")
        return

    # --- 2. Load Model, Index, and Metadata ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = SentenceTransformer(MODEL_PATH, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading index from: {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"Loading metadata from: {meta_path}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # --- 3. Embed Query and Normalize if Necessary ---
    print(f"\nEmbedding query: '{query_text}'")
    query_embedding = model.encode([query_text], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy().astype('float32')

    if index_format == 'cypherbench':
        # Normalize the query vector for Inner Product (cosine similarity) search
        norm = np.linalg.norm(query_embedding_np)
        if norm > 0:
            query_embedding_np = query_embedding_np / norm
        print("Query vector normalized for cosine similarity search.")

    # --- 4. Execute Search ---
    print(f"Searching for top {k} similar values...")
    distances, indices = index.search(query_embedding_np, k)

    # --- 5. Parse and Display Results based on Format ---
    print("\n--- Search Results ---")
    if not indices.size:
        print("No results found.")
        return

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        
        distance = distances[0][i]
        print(f"\nRank {i+1}:")

        if index_format == 'cypherbench':
            if idx < len(metadata):
                result_meta = metadata[idx]
                print(f"  - Similarity Score (L2 Distance): {distance:.4f}")
                print(f"  - Matched Value: '{result_meta.get('value')}'")
                print(f"  - Node Label: '{result_meta.get('node_label')}'")
                print(f"  - Attribute Name: '{result_meta.get('attribute_name')}'")
                print(f"  - Node ID: {result_meta.get('node_id')}")
            else:
                print(f"  - Metadata index {idx} out of bounds.")

        elif index_format == 'mtq':
            if idx < len(metadata):
                result_meta = metadata[idx]
                print(f"  - Similarity Score (L2 Distance): {distance:.4f}")
                print(f"  - Matched Value: '{result_meta.get('value')}'")
                print(f"  - Node Label: '{result_meta.get('node_label')}'")
                print(f"  - Attribute Name: '{result_meta.get('attribute_name')}'")
                print(f"  - Node ID: {result_meta.get('node_id')}")
            else:
                print(f"  - Metadata index {idx} out of bounds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search for similar attribute values using a FAISS index. Uses predefined lists to determine format.")
    parser.add_argument("schema_name", type=str, help="The name of the schema to search in (e.g., 'bloom50', 'soccer').")
    parser.add_argument("query", type=str, help="The text value to search for.")
    parser.add_argument("-k", type=int, default=10, help="The number of top results to return (default: 10).")

    args = parser.parse_args()
    search(args.schema_name, args.query, args.k)
