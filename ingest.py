"""Step 2: load the CSVs into SQLite (for SQL questions) and ChromaDB (for RAG over reviews)."""
import sqlite3

import chromadb
import pandas as pd

from config import CHROMA_DIR, COLLECTION, DATA_DIR, DB_PATH
from embeddings import get_embedding_function


def load_frames():
    products = pd.read_csv(DATA_DIR / "products.csv")
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    reviews = pd.read_csv(DATA_DIR / "reviews.csv")
    reviews["sentiment"] = reviews["rating"].apply(
        lambda r: "positive" if r >= 4 else ("neutral" if r == 3 else "negative")
    )
    return products, sales, reviews


def build_sqlite(products, sales, reviews):
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    products.to_sql("products", con, index=False)
    sales.to_sql("sales", con, index=False)
    reviews.to_sql("reviews", con, index=False)
    con.commit()
    con.close()
    print(f"SQLite ready: {len(products)} products, {len(sales)} sales rows, {len(reviews)} reviews")


def build_chroma(products, reviews):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(
        COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    merged = reviews.merge(products[["product_id", "product_name"]], on="product_id")
    batch = 100
    for start in range(0, len(merged), batch):
        chunk = merged.iloc[start : start + batch]
        collection.add(
            ids=[f"R{rid}" for rid in chunk["review_id"]],
            documents=chunk["review_text"].tolist(),
            metadatas=[
                {
                    "product_name": r.product_name,
                    "rating": int(r.rating),
                    "sentiment": r.sentiment,
                    "review_date": r.review_date,
                }
                for r in chunk.itertuples()
            ],
        )
    print(f"ChromaDB ready: {collection.count()} reviews embedded")


if __name__ == "__main__":
    products, sales, reviews = load_frames()
    build_sqlite(products, sales, reviews)
    build_chroma(products, reviews)
