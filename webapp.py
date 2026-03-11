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

# 1. OPTYMALIZACJA: Baza ładuje się tylko RAZ, a nie przy każdym pytaniu
@st.cache_resource(show_spinner="Ładowanie bazy wiedzy...")
def load_and_prepare_db():
    embeddings = HuggingFaceEmbeddings()
    documents = []
    folder = "documents"

    # Zabezpieczenie: sprawdza czy folder istnieje i czy są w nim PDF-y
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
    
    db = Chroma.from_documents(texts, embeddings)
    return db

# Uruchomienie bazy z pamięci podręcznej
db = load_and_prepare_db()

# Konfiguracja klucza OpenAI (najpierw szuka w chmurze Streamlit, potem w pliku .env)
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

    # Jeśli baza z jakiegoś powodu się nie załadowała, przerwij
    if db is None:
        st.error("Baza dokumentów jest pusta. Bot nie ma z czego czytać.")
        st.stop()

    # 2. NAPRAWA WYSZUKIWANIA: Czysty prompt i usuwanie duplikatów
    results = db.similarity_search(prompt, k=6)

    unique_texts = []
    for r in results:
        if r.page_content not in unique_texts:
            unique_texts.append(r.page_content)

    context = "\n\n---\n\n".join(unique_texts)

    # --- PANEL DIAGNOSTYCZNY DLA DYSPOZYTORA ---
    with st.expander("🔍 Podgląd z maszynowni (Co dokładnie widzi bot?)"):
        if not context.strip():
            st.error("UWAGA: Baza wektorowa zwróciła PUSTY tekst.")
        else:
            st.info("Bot otrzymał do analizy następujący tekst z PDF-a:")
            st.write(context)
    # -------------------------------------------

    # Prompt dla modelu językowego AI (zabezpieczony przed halucynacjami)
    full_prompt = f"""
    Jesteś profesjonalnym i pomocnym asystentem studenta.
    Twoim zadaniem jest odpowiadać na pytania na podstawie PONIŻSZYCH fragmentów regulaminu uczelni.

    ZASADY:
    1. Opieraj się WYŁĄCZNIE na dostarczonym tekście.
    2. Jeśli w tekście są podane konkretne kwoty, progi lub średnie, zacytuj je.
    3. Jeśli odpowiedź na pytanie NIE znajduje się w poniższych fragmentach, napisz dokładnie:
    "Przepraszam, ale nie znalazłem tej informacji w aktualnym regulaminie. Skontaktuj się z dziekanatem."
    4. Nie wymyślaj własnych odpowiedzi, nie korzystaj z wiedzy ogólnej.

    FRAGMENTY REGULAMINU:
    {context}

    PYTANIE STUDENTA:
    {prompt}

    ODPOWIEDŹ:
    """

    # Odpytanie modelu OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.choices[0].message.content

    # Wyświetlenie i zapisanie odpowiedzi
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})