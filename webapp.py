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

# --- TRYB PRO: Ukrywanie domyślnego menu i stopki Streamlit ---
st.set_page_config(page_title="SAM - Studencki Asystent", page_icon="🎓")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# -------------------------------------------------------------

# --- PANEL BOCZNY (WYBÓR UCZELNI I PRAWA AUTORSKIE) ---
with st.sidebar:
    st.markdown("### 🎓 SAM")
    st.markdown("Studencki Asystent oparty na zaawansowanej architekturze RAG.")
    
    st.markdown("---")
    st.markdown("#### Konfiguracja")
    lista_uczelni = ["merito", "uw", "uj"]
    wybrana_uczelnia = st.selectbox("Wybierz swoją uczelnię:", lista_uczelni)
    
    st.markdown("---")
    st.caption("Copyright (c) 2026 Krzysztof Adamiak. All rights reserved.")
# --------------------------------------

st.title(f"🎓 SAM – Studencki Asystent ({wybrana_uczelnia.upper()})")

# Pobranie klucza na samym starcie
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Brak klucza API OpenAI! Skonfiguruj plik .env lub Streamlit Secrets.")
    st.stop()

# 1. Ładowanie bazy z optymalizacją pod wiele uczelni i twardym formatowaniem
@st.cache_resource(show_spinner="Ładowanie dokumentów uczelni...")
def load_and_prepare_db(_api_key):
    embeddings = OpenAIEmbeddings(api_key=_api_key, model="text-embedding-3-small")
    documents = []
    base_folder = "documents"

    if os.path.exists(base_folder):
        for uczelnia_folder in os.listdir(base_folder):
            uczelnia_path = os.path.join(base_folder, uczelnia_folder)
            
            if os.path.isdir(uczelnia_path):
                for filename in os.listdir(uczelnia_path):
                    if filename.endswith(".pdf"):
                        file_path = os.path.join(uczelnia_path, filename)
                        loader = PyPDFLoader(file_path)
                        loaded_docs = loader.load()
                        
                        # Wstrzykiwanie metadanych: pancerne małe litery (.lower())
                        for doc in loaded_docs:
                            doc.metadata["uczelnia"] = uczelnia_folder.lower()
                        
                        documents.extend(loaded_docs)
    else:
        st.error(f"Brak głównego folderu '{base_folder}' z dokumentami!")
        return None
    
    if not documents:
        st.warning("Nie znaleziono żadnych plików PDF w podfolderach uczelni.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    # BOMBA NA CACHE: v3 w nazwie kolekcji zmusza Chromę do zbudowania bazy od zera
    db = Chroma.from_documents(texts, embeddings, collection_name="multi_uni_radar_v3")
    return db

# Uruchomienie bazy
db = load_and_prepare_db(api_key)
client = OpenAI(api_key=api_key)

# Inicjalizacja pamięci rozmowy czatu
if "messages" not in st.session_state:
    st.session_state.messages = []

# Czyszczenie historii czatu przy zmianie uczelni
if "ostatnia_uczelnia" not in st.session_state:
    st.session_state.ostatnia_uczelnia = wybrana_uczelnia

if st.session_state.ostatnia_uczelnia != wybrana_uczelnia:
    st.session_state.messages = []
    st.session_state.ostatnia_uczelnia = wybrana_uczelnia
    st.rerun()

# Wyświetlanie historii czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Główne pole wprowadzania pytań
if prompt := st.chat_input("Zadaj pytanie (np. jaka jest minimalna średnia na stypendium rektora?)"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    if db is None:
        st.error("Baza dokumentów jest pusta. Bot nie ma z czego czytać.")
        st.stop()

    # --- ZBIERANIE KONTEKSTU Z FILTREM UCZELNI (z wymuszeniem .lower()) ---
    results = db.max_marginal_relevance_search(
        prompt, 
        k=12, 
        fetch_k=30,
        filter={"uczelnia": wybrana_uczelnia.lower()}
    )
    
    unique_texts = []
    for r in results:
        if r.page_content not in unique_texts:
            unique_texts.append(r.page_content)
    context = "\n\n---\n\n".join(unique_texts)

    # --- RENTGEN (Do debugowania) ---
    with st.expander("🔍 Podgląd radaru (Co znalazł system RAG?)"):
        st.write(context if context else "Pusto! Radar nic nie pobrał z bazy.")

    # --- NATYWNA PAMIĘĆ OPENAI ---
    system_prompt = f"""
    Jesteś profesjonalnym i pomocnym asystentem studenta uczelni {wybrana_uczelnia.upper()}.
    Odpowiadasz na pytania na podstawie PONIŻSZYCH fragmentów regulaminu tej uczelni.

    ZASADY:
    1. Opieraj się WYŁĄCZNIE na dostarczonym tekście.
    2. Jeśli w tekście są podane konkretne kwoty, progi lub średnie, zacytuj je.
    3. Formułkę "Przepraszam, ale nie znalazłem tej informacji w aktualnym regulaminie. Skontaktuj się z dziekanatem." stosuj WYŁĄCZNIE wtedy, gdy we fragmentach nie ma absolutnie żadnej odpowiedzi. Jeśli udzielasz jakiejkolwiek merytorycznej odpowiedzi, NIGDY nie doklejaj tej formułki.
    4. SŁOWNIK: "średnia" to w regulaminach "Łączna liczba punktów".
    5. LOGIKA STYPENDIÓW: Tabela z kwotami Stypendium Rektora jest uniwersalna dla wszystkich kierunków! Kiedy student pyta o kwotę, weź jego średnią z historii rozmowy i od razu odczytaj kwotę z tej tabeli.
    6. LOGIKA ODLEGŁOŚCI: Jeśli kwota dofinansowania zależy od kilometrów (np. koszty podróży Erasmus), zapytaj studenta o dokładną odległość w kilometrach, zamiast podawać stawkę w ciemno.
    7. Zwracaj szczególną uwagę na § 9 ust. 4 regulaminu – stypendia są wypłacane regularnie co miesiąc aż do czerwca włącznie, a daty grudzień/maj są jedynie terminami ostatecznymi dla skumulowanych wypłat z początku semestrów.
    8. Analizuj intencję użytkownika. Pytania o to "jak zostać studentem" traktuj szeroko, jako zapytania o proces rekrutacji i wymagane w nim dokumenty.

    FRAGMENTY REGULAMINU:
    {context}
    """

    api_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in st.session_state.messages[-6:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=api_messages
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})