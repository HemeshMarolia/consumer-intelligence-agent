"""Streamlit app: ask questions and analyse product feedback.  Run: streamlit run app.py"""
import streamlit as st

from config import DB_PATH
from sql_tool import run_sql

st.set_page_config(page_title="Consumer Intelligence Agent", layout="wide")
st.title("AI-Powered Consumer Intelligence Agent")

if not DB_PATH.exists():
    st.error("No data found. Run `python generate_data.py` and then `python ingest.py` first.")
    st.stop()

# Imported after the check so a missing database gives a friendly message instead of a crash.
from agent import analyze_product, answer_question  # noqa: E402

tab_ask, tab_feedback, tab_data = st.tabs(["Ask a question", "Feedback analysis", "Data"])

# ---------------- Tab 1: ask anything ----------------
with tab_ask:
    examples = [
        "Which product has the highest total revenue?",
        "What do customers complain about for the Vitamin C Serum?",
        "Which product has the lowest average rating and why are customers unhappy with it?",
    ]
    choice = st.selectbox("Try an example", [""] + examples)
    question = st.text_input("Your question", value=choice)

    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            result = answer_question(question)
        st.caption(f"Tool used: {result['route']}")
        st.markdown(result["answer"])

        if result["sql"]:
            with st.expander("SQL used"):
                st.code(result["sql"], language="sql")
            df = result["df"]
            if df is not None and not df.empty:
                st.dataframe(df, width="stretch")
                if df.shape[1] == 2 and 2 <= len(df) <= 30 and df.iloc[:, 1].dtype.kind in "if":
                    st.bar_chart(df.set_index(df.columns[0]))
        if result["hits"]:
            with st.expander("Review evidence (sources)"):
                for h in result["hits"]:
                    st.markdown(f"**[{h['id']}]** {h['product_name']} - {h['rating']} stars - {h['text']}")

# ---------------- Tab 2: feedback analysis agent ----------------
with tab_feedback:
    products = run_sql("SELECT product_name FROM products ORDER BY product_name")["product_name"].tolist()
    product = st.selectbox("Product", products)
    if st.button("Analyse complaints"):
        with st.spinner("Classifying complaints and drafting recommendations..."):
            out = analyze_product(product)
        st.subheader(f"Complaint themes ({out['n_reviews']} negative reviews)")
        if not out["themes"].empty:
            st.dataframe(out["themes"], width="stretch")
            st.bar_chart(out["themes"].set_index("theme")["complaints"])
        st.subheader("Prioritized recommendations")
        st.markdown(out["recommendations"])

# ---------------- Tab 3: raw data ----------------
with tab_data:
    table = st.selectbox("Table", ["products", "sales", "reviews"])
    st.dataframe(run_sql(f"SELECT * FROM {table} LIMIT 200"), width="stretch")
