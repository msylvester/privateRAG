#!/usr/bin/env python3
import sys
import openai
import pandas as pd
import numpy as np
import argparse

def get_embedding(text, engine="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=engine  # Changed from engine to model parameter
    )
    return response["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    parser = argparse.ArgumentParser(description='Perform RAG using OpenAI embeddings on a CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file containing documents.')
    parser.add_argument('--text_column', default='text', help='Column name that contains text (default: text).')
    args = parser.parse_args()

    # Load CSV file
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if args.text_column not in df.columns:
        print(f"Column '{args.text_column}' not found in CSV file.")
        sys.exit(1)

    # Compute embeddings for each document
    embeddings = []
    print("Computing embeddings for documents...")
    for idx, text in enumerate(df[args.text_column]):
        print(f"Processing document {idx+1}/{len(df)}: {text[:30]}...")
        embedding = get_embedding(text)
        embeddings.append(embedding)

    # Get query
    query = input("Enter your query: ")
    query_embedding = get_embedding(query)

    # Compute cosine similarities
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]

    # Retrieve the best match
    best_idx = int(np.argmax(similarities))
    best_text = df[args.text_column].iloc[best_idx]

    print("\nBest matching document:")
    print(best_text)

if __name__ == "__main__":
    main()
