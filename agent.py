"""The agent layer: a router that picks the right tool, and a feedback-analysis workflow."""
import json

import pandas as pd

from llm import ask_llm
from rag import answer_rag
from sql_tool import answer_sql, run_sql

THEMES = [
    "packaging", "fragrance", "texture", "breakouts_irritation",
    "dryness", "price", "delivery", "ineffective", "other",
]


# ---------- 1. Router: decide which tool answers the question ----------

def route_question(question: str) -> str:
    """Return 'sql', 'rag' or 'both'."""
    reply = ask_llm(
        "Classify the question for a skincare analytics assistant.\n"
        "- sql: needs numbers from sales or ratings tables (revenue, units, counts, averages, rankings)\n"
        "- rag: asks what customers say, why they complain, themes in review text\n"
        "- both: needs numbers AND customer reasons\n"
        f"Question: {question}\nReply with exactly one word: sql, rag or both."
    ).strip().lower()
    for choice in ("both", "sql", "rag"):
        if choice in reply:
            return choice
    return "both"


def answer_question(question: str) -> dict:
    route = route_question(question)
    result = {"route": route, "sql": None, "df": None, "hits": [], "answer": ""}
    sql_part = answer_sql(question) if route in ("sql", "both") else None
    rag_part = answer_rag(question) if route in ("rag", "both") else None

    if sql_part:
        result.update(sql=sql_part["sql"], df=sql_part["df"])
    if rag_part:
        result["hits"] = rag_part["hits"]

    if route == "sql":
        result["answer"] = sql_part["answer"]
    elif route == "rag":
        result["answer"] = rag_part["answer"]
    else:
        result["answer"] = ask_llm(
            f"Question: {question}\n\nNumbers from the sales database:\n{sql_part['answer']}\n\n"
            f"What customers say (cite review ids as given):\n{rag_part['answer']}\n\n"
            "Combine both into one short answer. Do not add facts that are not above."
        )
    return result


# ---------- 2. Feedback agent: classify complaints -> themes -> recommendations ----------

def _parse_json_list(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def classify_reviews(reviews: pd.DataFrame, batch_size: int = 15) -> pd.DataFrame:
    """Ask the LLM to label each review with ONE theme. Returns reviews plus a 'theme' column."""
    labels = {}
    for start in range(0, len(reviews), batch_size):
        chunk = reviews.iloc[start : start + batch_size]
        lines = "\n".join(f"{r.review_id}: {r.review_text}" for r in chunk.itertuples())
        reply = ask_llm(
            f"Classify each complaint into exactly one theme from: {THEMES}.\n"
            'Reply with ONLY a JSON list like [{"id": 12, "theme": "packaging"}].\n\n' + lines
        )
        for item in _parse_json_list(reply):
            try:
                theme = item["theme"] if item["theme"] in THEMES else "other"
                labels[int(item["id"])] = theme
            except (KeyError, TypeError, ValueError):
                continue
    out = reviews.copy()
    out["theme"] = out["review_id"].map(labels).fillna("other")
    return out


def analyze_product(product_name: str) -> dict:
    """Full workflow for one product: pull complaints, classify, count, then recommend."""
    safe_name = product_name.replace("'", "''")
    negatives = run_sql(
        "SELECT r.review_id, r.rating, r.review_text FROM reviews r "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE p.product_name = '{safe_name}' AND r.sentiment = 'negative' LIMIT 60"
    )
    if negatives.empty:
        return {"themes": pd.DataFrame(), "recommendations": "No negative reviews found.", "n_reviews": 0}

    labelled = classify_reviews(negatives)
    themes = (
        labelled.groupby("theme")
        .agg(complaints=("review_id", "count"), example=("review_text", "first"))
        .sort_values("complaints", ascending=False)
        .reset_index()
    )
    themes["share_pct"] = (100 * themes["complaints"] / themes["complaints"].sum()).round(1)

    trend = run_sql(
        "SELECT s.sale_month, SUM(s.units) AS units FROM sales s "
        "JOIN products p ON p.product_id = s.product_id "
        f"WHERE p.product_name = '{safe_name}' GROUP BY s.sale_month ORDER BY s.sale_month"
    )
    recommendations = ask_llm(
        f"Product: {product_name}\n\nComplaint themes (from {len(labelled)} negative reviews):\n"
        f"{themes[['theme', 'complaints', 'share_pct', 'example']].to_string(index=False)}\n\n"
        f"Monthly units sold:\n{trend.to_string(index=False)}\n\n"
        "Write 3-5 prioritized product improvement recommendations. For each give: "
        "Priority (High/Medium/Low), the action, and the evidence (theme share or sales trend). "
        "Use only the data above.",
        temperature=0.2,
    )
    return {"themes": themes, "recommendations": recommendations, "n_reviews": len(labelled)}
