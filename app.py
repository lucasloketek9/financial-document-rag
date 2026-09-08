import streamlit as st
from ask import answer

st.set_page_config(page_title="Financial Document RAG Assistant", page_icon="📊")

st.title("📊 Financial Document RAG Assistant")
st.markdown(
    "Ask questions about the 10-K annual filings of **JPMorgan**, "
    "**Bank of America**, and **Wells Fargo**. "
    "Answers are grounded in the actual SEC filings and cite their sources."
)

# Some example questions to guide the user
st.markdown("**Try asking:**")
examples = [
    "How do these banks approach credit risk?",
    "What are JPMorgan's main risk factors?",
    "How much did Bank of America pay in income taxes?",
]
for ex in examples:
    st.markdown(f"- *{ex}*")

question = st.text_input("Your question:", placeholder="Ask about the banks' filings...")

if question:
    with st.spinner("Searching filings and generating answer..."):
        ans, hits = answer(question)
    st.markdown("### Answer")
    st.write(ans)
    st.markdown("### Sources retrieved")
    for chunk, src, score in hits:
        st.markdown(f"- **{src}** (similarity: {score:.3f})")