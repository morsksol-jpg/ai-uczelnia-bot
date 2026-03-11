import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Ładowanie zmiennych środowiskowych (dla testów lokalnych)
load_dotenv()

st.title("🎓 Bot uczelni – przewodnik po biurokracji")

# 1. Ładowanie bazy z optymalizacją (tylko raz) i polskim radarem
@st.cache_resource(show_spinner="Ładowanie bazy wiedzy...")
def load_and_prepare_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    documents = []
    folder = "documents"

    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
    else:
        st.error(f"Brak folderu '{folder}' z dokumentami!")
        return None
    
    if not documents:
        st.warning("Nie znaleziono żadnych plików PDF w folderze documents.")
        return None

    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    # Wymuszamy stworzenie czystego magazynu dla polskiego modelu
    db = Chroma.from_documents(texts, embeddings, collection_name="nowy_radar_pl_v1")
    return db

# Uruchomienie bazy z pamięci podręcznej
db = load_and_prepare_db()

# Konfiguracja klucza OpenAI
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Brak klucza API OpenAI! Skonfiguruj plik .env lub Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# Inicjalizacja pamięci rozmowy czatu
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie historii czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Główne pole wprowadzania pytań
if prompt := st.chat_input("Zadaj pytanie (np. jaka jest minimalna średnia na stypendium rektora?)"):

    # Zapisz pytanie użytkownika do historii
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if db is None:
        st.error("Baza dokumentów jest pusta. Bot nie ma z czego czytać.")
        st.stop()

    # 2. Szukanie kontekstu (z różnorodnością wyników MMR)
    results = db.max_marginal_relevance_search(prompt, k=5, fetch_k=20)

    unique_texts = []
    for r in results:
        if r.page_content not in unique_texts:
            unique_texts.append(r.page_content)

    context = "\n\n---\n\n".join(unique_texts)

    # 3. Prompt dla modelu językowego AI
    full_prompt = f"""
    Jesteś profesjonalnym i pomocnym asystentem studenta.
    Twoim zadaniem jest odpowiadać na pytania na podstawie PONIŻSZYCH fragmentów regulaminu ucz