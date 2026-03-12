import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Ładowanie zmiennych środowiskowych
load_dotenv()

# --- TRYB PRO: Ukrywanie domyślnego menu i stopki Streamlit ---
st.set_page_config(page_title="Bot Uczelni", page_icon="🎓")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# -------------------------------------------------------------

st.title("🎓 Bot uczelni – przewodnik po biurokracji")

# Pobranie klucza na samym starcie
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Brak klucza API OpenAI! Skonfiguruj plik .env lub Streamlit Secrets.")
    st.stop()

# 1. Ładowanie bazy z optymalizacją - radar pracuje na serwerach OpenAI
@st.cache_resource(show_spinner="Ładowanie dokumentów uczelni...")
def load_and_prepare_db(_api_key):
    embeddings = OpenAIEmbeddings(api_key=_api_key, model="text-embedding-3-small")
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
    
    # Tworzymy czysty magazyn
    db = Chroma.from_documents(texts, embeddings, collection_name="openai_radar_v2")
    return db

# Uruchomienie bazy
db = load_and_prepare_db(api_key)
client = OpenAI(api_key=api_key)

# Inicjalizacja pamięci rozmowy czatu i modułu głosowego
if "messages" not in st.session_state:
    st.session_state.messages = []

# Blokada, żeby ten sam plik audio nie przetwarzał się w kółko
if "processed_audio" not in st.session_state:
    st.session_state.processed_audio = None

# Wyświetlanie historii czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- MODUŁ WEJŚCIA (GŁOS LUB TEKST) ---
user_question = None

# 1. Opcja Głosowa (Mikrofon wbudowany w Streamlit)
st.info("💡 Tryb dostępności: Zezwól na dostęp do mikrofonu, aby zadać pytanie głosowo.")
audio_val = st.audio_input("Nagraj swoje pytanie")

# Jeśli nagrano nowy głos, tłumacz na tekst
if audio_val and audio_val != st.session_state.processed_audio:
    st.session_state.processed_audio = audio_val
    with st.spinner("Przetwarzam głos na tekst..."):
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_val
        )
        user_question = transcript.text

# 2. Opcja Tekstowa (Tradycyjny czat)
prompt = st.chat_input("Zadaj pytanie tekstowo...")
if prompt:
    user_question = prompt

# --- GŁÓWNA LOGIKA ODPOWIEDZI ---
if user_question:
    # Zapisz i wyświetl pytanie użytkownika
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if db is None:
        st.error("Baza dokumentów jest pusta. Bot nie ma z czego czytać.")
        st.stop()

    # Zbieranie kontekstu (Amnezja - Lekarstwo)
    user_queries = [msg["content"] for msg in st.session_state.messages[-4:] if msg["role"] == "user"]
    search_query = " ".join(user_queries)
    results = db.max_marginal_relevance_search(search_query, k=12, fetch_k=30)
    
    unique_texts = []
    for r in results:
        if r.page_content not in unique_texts:
            unique_texts.append(r.page_content)
    context = "\n\n---\n\n".join(unique_texts)

    # NATYWNA PAMIĘĆ OPENAI
    system_prompt = f"""
    Jesteś profesjonalnym i pomocnym asystentem studenta.
    Odpowiadasz na pytania na podstawie PONIŻSZYCH fragmentów regulaminu uczelni.

    ZASADY:
    1. Opieraj się WYŁĄCZNIE na dostarczonym tekście.
    2. Jeśli w tekście są podane konkretne kwoty, progi lub średnie, zacytuj je.
    3. Jeśli odpowiedź NIE znajduje się w poniższych fragmentach, napisz dokładnie: "Przepraszam, ale nie znalazłem tej informacji w aktualnym regulaminie. Skontaktuj się z dziekanatem."
    4. SŁOWNIK: "średnia" to w regulaminach "Łączna liczba punktów".
    5. LOGIKA STYPENDIÓW: Tabela z kwotami Stypendium Rektora jest uniwersalna dla wszystkich kierunków! Kiedy student pyta o kwotę, weź jego średnią z historii rozmowy i od razu odczytaj kwotę z tej tabeli.

    FRAGMENTY REGULAMINU:
    {context}
    """

    api_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in st.session_state.messages[-6:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Generowanie odpowiedzi AI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=api_messages
    )

    answer = response.choices[0].message.content

    # Wyświetlanie i czytanie odpowiedzi
    with st.chat_message("assistant"):
        st.markdown(answer)
        
        # --- MODUŁ LEKTORA (TTS) ---
        with st.spinner("🎙️ Przygotowuję odpowiedź głosową..."):
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice="nova", # Realistyczny, przyjemny głos żeński
                input=answer
            )
            # Odtwarzamy dźwięk automatycznie dzięki autoplay=True
            st.audio(audio_response.content, format="audio/mp3", autoplay=True)

    # Zapisz odpowiedź bota do historii
    st.session_state.messages.append({"role": "assistant", "content": answer})