import streamlit as st
from ask import retrieve
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
CHAT_MODEL = os.getenv("AZURE_CHAT_DEPLOYMENT")

st.set_page_config(page_title="Financial Document RAG Assistant", page_icon="📊")
st.title("📊 Financial Document RAG Assistant")
st.markdown(
    "Ask questions about the 10-K filings of **JPMorgan**, **Bank of America**, "
    "and **Wells Fargo**. Answers are grounded in the filings and remember the conversation."
)

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask about the banks' filings..."):
    # Show and store the user's question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant chunks for this question
    hits = retrieve(question, top_k=5)
    context = "\n\n".join(f"[Source: {src}]\n{chunk}" for chunk, src, score in hits)

    # Build the message list: system prompt + full history + retrieved context
    system_prompt = (
        "You are a financial analyst assistant answering questions about bank 10-K filings. "
        "Use the provided context to answer, cite which bank each fact comes from, and use the "
        "conversation history to understand follow-up questions. If the answer isn't in the "
        "context, say so."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.messages)  # full history for memory
    messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
            )
            answer_text = response.choices[0].message.content
            st.markdown(answer_text)
            with st.expander("Sources retrieved"):
                for chunk, src, score in hits:
                    st.markdown(f"- **{src}** (similarity: {score:.3f})")

    st.session_state.messages.append({"role": "assistant", "content": answer_text})