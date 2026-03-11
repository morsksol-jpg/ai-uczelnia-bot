import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

st.title("🎓 Bot uczelni – przewodnik po biurokracji")

# embeddings
embeddings = HuggingFaceEmbeddings()

# wczytanie dokumentów
documents = []
folder = "documents"

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder, filename))
        documents.extend(loader.load())

# dzielenie tekstu
text_splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

texts = text_splitter.split_documents(documents)

# baza wektorowa
db = Chroma.from_documents(texts, embeddings)

# klient OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# pamięć rozmowy
if "messages" not in st.session_state:
    st.session_state.messages = []

# wyświetlanie historii
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# pole pytania
if prompt := st.chat_input("Zadaj pytanie"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # wyszukiwanie fragmentów
    results = db.similarity_search(prompt, k=12)

    # filtrowanie załączników
    filtered = []

    for r in results:
        text = r.page_content.lower()

        if "załącznik" not in text and "wzór wniosku" not in text:
            filtered.append(r.page_content)

    context = "\n\n".join(filtered)

    # prompt dla AI
    full_prompt = f"""
Odpowiadaj WYŁĄCZNIE na podstawie fragmentów regulaminu.

Jeśli odpowiedź nie znajduje się w fragmentach napisz:
"Nie znalazłem informacji w regulaminie."

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