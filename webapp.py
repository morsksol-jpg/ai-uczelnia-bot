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

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Ścieżka do bazy wektorowej
documents = []
folder = "documents"

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder, filename))
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

texts = text_splitter.split_documents(documents)

db = Chroma.from_documents(texts, embeddings)

# Klient OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pamięć rozmowy
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie historii
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Pole pytania
if prompt := st.chat_input("Zadaj pytanie"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # wyszukiwanie fragmentów regulaminu
    results = db.similarity_search(prompt, k=6)

    context = "\n\n".join([r.page_content for r in results])
    st.write("Znalezione fragmenty:", context)

    # prompt dla AI
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