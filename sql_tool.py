"""Natural language -> SQL, with a safety check and a read-only database connection."""
import re
import sqlite3

import pandas as pd

from config import DB_PATH
from llm import ask_llm

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


def connect():
    """Open the database read-only, so even a bad query cannot change data."""
    return sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)


def is_safe_sql(sql: str):
    """Allow exactly one SELECT (or WITH ... SELECT) statement. Returns (ok, reason)."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False, "Empty query"
    if ";" in s:
        return False, "Only one statement is allowed"
    if "--" in s or "/*" in s:
        return False, "Comments are not allowed"
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return False, "Only SELECT queries are allowed"
    if FORBIDDEN.search(s):
        return False, "Query contains a forbidden keyword"
    return True, ""


def get_schema() -> str:
    """Table definitions plus the exact values of key columns, so the LLM stops guessing."""
    con = connect()
    ddl = "\n".join(r[0] for r in con.execute("SELECT sql FROM sqlite_master WHERE type='table'"))
    hints = []
    for table, col in [
        ("products", "product_name"),
        ("products", "category"),
        ("sales", "channel"),
        ("sales", "region"),
        ("reviews", "sentiment"),
    ]:
        values = [r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM {table} ORDER BY 1")]
        hints.append(f"{table}.{col} values: {values}")
    months = [r[0] for r in con.execute("SELECT DISTINCT sale_month FROM sales ORDER BY 1")]
    hints.append(f"sales.sale_month values (format YYYY-MM): {months}")
    con.close()
    return ddl + "\n\n" + "\n".join(hints)


def clean_sql(text: str) -> str:
    """Remove markdown code fences the LLM sometimes adds."""
    text = text.strip()
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def generate_sql(question: str, previous_sql: str = "", error: str = "") -> str:
    prompt = f"""You are a SQLite expert. Write ONE SQLite query that answers the question.

Database schema:
{get_schema()}

Rules:
- Return ONLY the SQL query, no explanation and no markdown.
- Use SELECT only. Use exact values from the lists above (e.g. exact product names).
- Join products to sales or reviews on product_id when you need product names or categories.
- revenue is in INR. Round money to 2 decimals.
- Add LIMIT 50 unless the question asks for a total or a single value.
"""
    if error:
        prompt += f"\nYour previous query failed.\nPrevious query: {previous_sql}\nError: {error}\nFix it.\n"
    prompt += f"\nQuestion: {question}\nSQL:"
    return clean_sql(ask_llm(prompt))


def run_sql(sql: str) -> pd.DataFrame:
    con = connect()
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def answer_sql(question: str) -> dict:
    """Generate SQL, check it, run it (one retry on error), then explain the result in words."""
    sql = generate_sql(question)
    df, error = None, ""
    for _ in range(2):
        ok, reason = is_safe_sql(sql)
        if not ok:
            return {"sql": sql, "df": None, "answer": f"I refused to run that query: {reason}.", "error": reason}
        try:
            df = run_sql(sql)
            error = ""
            break
        except Exception as exc:  # bad column name, syntax error, etc.
            error = str(exc)
            sql = generate_sql(question, previous_sql=sql, error=error)
    if df is None:
        return {"sql": sql, "df": None, "answer": f"The query failed: {error}", "error": error}

    table_text = df.head(30).to_string(index=False)
    answer = ask_llm(
        f"Question: {question}\n\nSQL result:\n{table_text}\n\n"
        "Answer the question in 2-4 sentences using ONLY the numbers in the result above. "
        "Do not invent numbers."
    )
    return {"sql": sql, "df": df, "answer": answer, "error": ""}
