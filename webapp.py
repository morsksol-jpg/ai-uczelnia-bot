import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.title("🎓 Bot uczelni – przewodnik po biurokracji")

embeddings = HuggingFaceEmbeddings()

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# pamięć rozmowy
if "messages" not in st.session_state:
    st.session_state.messages = []

# wyświetlanie historii rozmowy
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# pole wpisania pytania
if prompt := st.chat_input("Zadaj pytanie"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    results = db.similarity_search(prompt, k=6)

    context = "\n\n".join([r.page_content for r in results])

    full_prompt = f"""
Na podstawie poniższych fragmentów regulaminów uczelni odpowiedz na pytanie studenta.

Fragmenty regulaminu:
{context}

Pytanie studenta:
{prompt}

Odpowiedź:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})