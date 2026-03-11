import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.title("🎓 Bot uczelni – przewodnik po biurokracji")

embeddings = HuggingFaceEmbeddings()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

db_path = "vector_db"

if os.path.exists(db_path):
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    documents = []
    folder = "documents"

    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, filename))
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=db_path
    )
    db.persist()

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
    st.write(context)
    full_prompt = f"""
Odpowiadaj WYŁĄCZNIE na podstawie podanych fragmentów regulaminu uczelni.

Jeśli odpowiedź nie znajduje się w fragmentach, napisz:
"Nie znalazłem informacji w regulaminie."

Nie wymyślaj informacji.

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