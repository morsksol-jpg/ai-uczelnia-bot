import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Ładowanie zmiennych środowiskowych
load_dotenv()

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="SAM - Studencki Asystent Merito", page_icon="🎓")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎓 SAM")
    st.markdown("Studencki Asystent Merito oparty na architekturze RAG.")
    st.markdown("---")
    st.caption("Copyright (c) 2026 Krzysztof Adamiak. All rights reserved.")

st.title("🎓 SAM – Studencki Asystent Merito")

# Pobranie klucza API
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Brak klucza API OpenAI! Skonfiguruj .env lub Streamlit Secrets.")
    st.stop()


# ==============================
# FUNKCJA ŁADOWANIA BAZY
# ==============================

@st.cache_resource(show_spinner="Ładowanie dokumentów uczelni...")
def load_and_prepare_db(_api_key):

    embeddings = OpenAIEmbeddings(
        api_key=_api_key,
        model="text-embedding-3-small"
    )

    persist_directory = "vector_db"

    # Jeśli baza istnieje – wczytaj ją
    if os.path.exists(persist_directory):

        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        return db

    # Jeśli baza nie istnieje – buduj ją z PDF
    documents = []
    folder = "documents"

    if not os.path.exists(folder):
        st.error(f"Brak folderu '{folder}' z dokumentami!")
        return None

    for filename in os.listdir(folder):

        if filename.endswith(".pdf"):

            file_path = os.path.join(folder, filename)

            loader = PyPDFLoader(file_path)

            documents.extend(loader.load())

    if not documents:
        st.warning("Nie znaleziono plików PDF w folderze documents.")
        return None

    # Chunkowanie dokumentów
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = text_splitter.split_documents(documents)

    # Tworzenie bazy wektorowej
    db = Chroma.from_documents(
        texts,
        embeddings,
        collection_name="openai_radar_v2",
        persist_directory=persist_directory
    )

    db.persist()

    return db


# ==============================
# URUCHOMIENIE SYSTEMU
# ==============================

db = load_and_prepare_db(api_key)

client = OpenAI(api_key=api_key)

# Historia czatu
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlenie historii
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==============================
# OBSŁUGA PYTAŃ
# ==============================

if prompt := st.chat_input("Zadaj pytanie (np. jaka jest minimalna średnia na stypendium rektora?)"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if db is None:
        st.error("Baza dokumentów jest pusta.")
        st.stop()

    # Wyszukiwanie kontekstu
    results = db.max_marginal_relevance_search(
        prompt,
        k=8,
        fetch_k=20
    )

    unique_texts = []

    for r in results:
        if r.page_content not in unique_texts:
            unique_texts.append(r.page_content)

    context = "\n\n---\n\n".join(unique_texts)

    system_prompt = f"""
Jesteś profesjonalnym asystentem studenta.

Odpowiadasz WYŁĄCZNIE na podstawie fragmentów regulaminu uczelni.

ZASADY:
1. Jeśli odpowiedź znajduje się w tekście – podaj ją.
2. Jeśli nie ma informacji – napisz:
"Przepraszam, ale nie znalazłem tej informacji w regulaminie. Skontaktuj się z dziekanatem."

FRAGMENTY REGULAMINU:

{context}
"""

    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in st.session_state.messages[-6:]:
        api_messages.append(msg)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=api_messages
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})