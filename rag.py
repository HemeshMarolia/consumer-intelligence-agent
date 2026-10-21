"""RAG over customer reviews: retrieve the most relevant reviews, answer with citations."""
import chromadb

from config import CHROMA_DIR, COLLECTION
from embeddings import get_embedding_function
from llm import ask_llm


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION, embedding_function=get_embedding_function())


def retrieve(question: str, k: int = 8, product_name: str = None, sentiment: str = None) -> list:
    """Return the k most similar reviews, optionally filtered by product and/or sentiment."""
    conditions = []
    if product_name:
        conditions.append({"product_name": product_name})
    if sentiment:
        conditions.append({"sentiment": sentiment})
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    res = get_collection().query(query_texts=[question], n_results=k, where=where)
    hits = []
    for rid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"id": rid, "text": doc, "similarity": round(1 - dist, 3), **meta})
    return hits


def answer_rag(question: str, k: int = 8) -> dict:
    hits = retrieve(question, k=k)
    if not hits:
        return {"answer": "No relevant reviews found.", "hits": []}
    context = "\n".join(
        f"[{h['id']}] ({h['product_name']}, {h['rating']} stars) {h['text']}" for h in hits
    )
    prompt = f"""You answer questions about customer reviews of skincare products.
Use ONLY the reviews below. After each claim, cite the supporting review ids like [R12].
If the reviews do not contain the answer, say so.

Reviews:
{context}

Question: {question}
Answer:"""
    return {"answer": ask_llm(prompt), "hits": hits}
