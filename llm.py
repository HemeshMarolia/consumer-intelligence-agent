"""One place that talks to the LLM, so it is easy to swap providers or stub in tests."""
from langchain_groq import ChatGroq

from config import GROQ_MODEL


def ask_llm(prompt: str, temperature: float = 0.0) -> str:
    """Send a single prompt and return the text reply. Needs GROQ_API_KEY in the environment."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=temperature)
    return llm.invoke(prompt).content
