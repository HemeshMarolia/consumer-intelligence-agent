"""Step 5: measure how accurate the text-to-SQL part is. Run: python eval.py

Each case has a question and a hand-written "gold" SQL query. We run both the gold SQL and the
SQL the LLM generates, then compare the results. This number goes on your resume.
"""
from sql_tool import generate_sql, is_safe_sql, run_sql

P = "p.product_name"
CASES = [
    ("What is the total revenue?",
     "SELECT ROUND(SUM(revenue), 2) FROM sales"),
    ("How many reviews are there in total?",
     "SELECT COUNT(*) FROM reviews"),
    ("Which product has the highest total revenue?",
     f"SELECT {P} FROM sales s JOIN products p ON s.product_id = p.product_id "
     f"GROUP BY {P} ORDER BY SUM(s.revenue) DESC LIMIT 1"),
    ("What is the total revenue by channel?",
     "SELECT channel, ROUND(SUM(revenue), 2) FROM sales GROUP BY channel"),
    ("How many negative reviews does GlowLab Vitamin C Serum have?",
     "SELECT COUNT(*) FROM reviews r JOIN products p ON r.product_id = p.product_id "
     f"WHERE {P} = 'GlowLab Vitamin C Serum' AND r.sentiment = 'negative'"),
    ("Which product has the lowest average rating?",
     f"SELECT {P} FROM reviews r JOIN products p ON r.product_id = p.product_id "
     f"GROUP BY {P} ORDER BY AVG(r.rating) ASC LIMIT 1"),
    ("Which region sold the most units?",
     "SELECT region FROM sales GROUP BY region ORDER BY SUM(units) DESC LIMIT 1"),
    ("How many units were sold through E-commerce in 2026-03?",
     "SELECT SUM(units) FROM sales WHERE channel = 'E-commerce' AND sale_month = '2026-03'"),
    ("Which product category has the highest average rating?",
     "SELECT category FROM reviews r JOIN products p ON r.product_id = p.product_id "
     "GROUP BY category ORDER BY AVG(r.rating) DESC LIMIT 1"),
    ("What are the top 3 products by units sold?",
     f"SELECT {P} FROM sales s JOIN products p ON s.product_id = p.product_id "
     f"GROUP BY {P} ORDER BY SUM(s.units) DESC LIMIT 3"),
]

UNSAFE = [
    "DROP TABLE sales",
    "SELECT 1; DELETE FROM sales",
    "UPDATE products SET price = 0",
    "SELECT * FROM sales -- comment",
    "PRAGMA table_info(sales)",
    "INSERT INTO reviews VALUES (1)",
]


def normalise(df):
    rows = []
    for row in df.itertuples(index=False):
        rows.append(tuple(round(v, 1) if isinstance(v, float) else v for v in row))
    return sorted(rows, key=str)


def flatten(rows):
    return {v for row in rows for v in row}


def answers_match(gold, got):
    """Correct answer: same number of rows as the gold result and every gold value is present.

    Extra columns are allowed (e.g. the LLM returns the product name AND its revenue), but
    returning more rows than expected, or missing a value, counts as wrong.
    """
    return len(got) == len(gold) and flatten(gold) <= flatten(got)


def main():
    strict = lenient = 0
    for question, gold_sql in CASES:
        gold = normalise(run_sql(gold_sql))
        generated_sql = generate_sql(question)
        try:
            ok, _ = is_safe_sql(generated_sql)
            got = normalise(run_sql(generated_sql)) if ok else []
        except Exception:
            got = []
        is_strict = got == gold
        is_lenient = is_strict or (bool(got) and answers_match(gold, got))
        strict += is_strict
        lenient += is_lenient
        print(f"{'PASS' if is_strict else ('OK+ ' if is_lenient else 'FAIL')}  {question}")
        if not is_strict:
            print(f"      generated: {generated_sql}")

    blocked = sum(not is_safe_sql(q)[0] for q in UNSAFE)
    n = len(CASES)
    print("\nPASS = identical result; OK+ = correct answer with extra columns; FAIL = wrong")
    print(f"Identical result: {strict}/{n} ({100 * strict / n:.0f}%)")
    print(f"Correct answer:   {lenient}/{n} ({100 * lenient / n:.0f}%)")
    print(f"Unsafe queries blocked: {blocked}/{len(UNSAFE)}")


if __name__ == "__main__":
    main()