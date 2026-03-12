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

    # --- LEKARSTWO NA AMNEZJĘ ---
    # Bierzemy 3 ostatnie wiadomości (żeby bot wiedział, o czym rozmawialiście przed chwilą)
    recent_history = st.session_state.messages[-3:]
    search_query = " ".join([msg["content"] for msg in recent_history])
    
    # Szukamy w bazie na podstawie pełnego kontekstu rozmowy
    results = db.max_marginal_relevance_search(search_query, k=5, fetch_k=20)
    # ----------------------------

    unique_texts = []
    for r in results:
        if r.page_content not in unique_texts:
            unique_texts.append(r.page_content)

    context = "\n\n---\n\n".join(unique_texts)

    # Prompt dla modelu AI
    full_prompt = f"""
    Jesteś profesjonalnym i pomocnym asystentem studenta.
    Twoim zadaniem jest odpowiadać na pytania na podstawie PONIŻSZYCH fragmentów regulaminu uczelni.

    ZASADY:
    1. Opieraj się WYŁĄCZNIE na dostarczonym tekście.
    2. Jeśli w tekście są podane konkretne kwoty, progi lub średnie, zacytuj je.
    3. Jeśli odpowiedź na pytanie NIE znajduje się w poniższych fragmentach, napisz dokładnie:
    "Przepraszam, ale nie znalazłem tej informacji w aktualnym regulaminie. Skontaktuj się z dziekanatem."
    4. Nie wymyślaj własnych odpowiedzi, nie korzystaj z wiedzy ogólnej.
    5. Odpowiadaj naturalnie, uprzejmie i konkretnie, biorąc pod uwagę kontekst całej rozmowy.

    FRAGMENTY REGULAMINU:
    {context}

    AKTUALNE PYTANIE STUDENTA:
    {prompt}

    ODPOWIEDŹ:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    # Zapisz odpowiedź bota do historii (ważne dla kolejnych pytań!)
    st.session_state.messages.append({"role": "assistant", "content": answer})