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

# --- KONFIGURACJA INTERFEJSU ---
st.set_page_config(page_title="SAM - Studencki Asystent", page_icon="🎓")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### 🎓 SAM")
    st.markdown("Profesjonalny asystent studenta (Wersja 2.0)")
    st.markdown("---")
    lista_uczelni = ["merito", "uw", "uj"]
    wybrana_uczelnia = st.selectbox("Wybierz swoją uczelnię:", lista_uczelni)
    st.markdown("---")
    st.caption("Copyright (c) 2026 Krzysztof Adamiak. All rights reserved.")

st.title(f"🎓 SAM – Studencki Asystent ({wybrana_uczelnia.upper()})")

# Pobranie klucza API
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Brak klucza API OpenAI!")
    st.stop()

# --- FUNKCJA ŁADOWANIA I OPTYMALIZACJI BAZY ---
@st.cache_resource(show_spinner="Inicjalizacja szybkiej bazy wiedzy...")
def load_and_prepare_db(_api_key):
    embeddings = OpenAIEmbeddings(api_key=_api_key, model="text-embedding-3-small")
    persist_directory = "vector_db"
    
    # Próba załadowania istniejącej bazy z dysku (SZYBKI START)
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings, 
            collection_name="sam_production_v2"
        )

    # Budowanie bazy od zera (tylko jeśli folder vector_db nie istnieje)
    documents = []
    base_folder = "documents"
    
    if os.path.exists(base_folder):
        for uczelnia_folder in os.listdir(base_folder):
            uczelnia_path = os.path.join(base_folder, uczelnia_folder)
            if os.path.isdir(uczelnia_path):
                for filename in os.listdir(uczelnia_path):
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(os.path.join(uczelnia_path, filename))
                        loaded_docs = loader.load()
                        for doc in loaded_docs:
                            # Zabezpieczenie przed wielkością liter w nazwach folderów
                            doc.metadata["uczelnia"] = uczelnia_folder.lower()
                        documents.extend(loaded_docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    
    # Zapisanie bazy na dysku
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        collection_name="sam_production_v2", 
        persist_directory=persist_directory
    )
    return db

db = load_and_prepare_db(api_key)
client = OpenAI(api_key=api_key)

# Inicjalizacja i czyszczenie pamięci
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ostatnia_uczelnia" not in st.session_state:
    st.session_state.ostatnia_uczelnia = wybrana_uczelnia

if st.session_state.ostatnia_uczelnia != wybrana_uczelnia:
    st.session_state.messages = []
    st.session_state.ostatnia_uczelnia = wybrana_uczelnia
    st.rerun()

# Wyświetlanie czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obsługa pytań
if prompt := st.chat_input("Zadaj pytanie dotyczące regulaminu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Wyszukiwanie z wysokim parametrem k dla lepszej skuteczności
    results = db.max_marginal_relevance_search(
        prompt, 
        k=25, 
        fetch_k=50,
        filter={"uczelnia": wybrana_uczelnia.lower()}
    )
    
    context = "\n\n---\n\n".join([r.page_content for r in results])

    system_prompt = f"""
    Jesteś profesjonalnym asystentem studenta uczelni {wybrana_uczelnia.upper()}.
    Odpowiadasz WYŁĄCZNIE na podstawie dostarczonych fragmentów regulaminu.

    ZASADY:
    1. Cytuj konkretne liczby, kwoty i progi.
    2. Jeśli odpowiedź nie istnieje w tekście, napisz: "Przepraszam, ale nie znalazłem tej informacji w aktualnym regulaminie. Skontaktuj się z dziekanatem."
    3. Nigdy nie dopisuj powyższej formułki, jeśli udzieliłeś merytorycznej odpowiedzi.
    4. Analizuj intencje – pytania o "ile osób" kojarz z sekcjami o liczebności grup lub komitetów założycielskich.
    5. LOGIKA BRAKU DANYCH (Stypendia): Jeśli student pyta o stypendium rektora, wyjaśnij, że jego wysokość zależy od ŁĄCZNEJ LICZBY PUNKTÓW, czyli: średnia ocen + punkty za osiągnięcia (naukowe, sportowe, artystyczne). 
    - Jeśli student podał tylko średnią, podaj kwotę dla tej średniej, ale dopytaj: "Czy masz dodatkowe punkty za osiągnięcia? Mogą one podnieść kwotę stypendium". 
    - Jeśli student podał i średnią, i punkty dodatkowe, DODAJ JE DO SIEBIE. Otrzymaną sumę odszukaj w tabeli "Łączna liczba punktów" i podaj mu ostateczną, konkretną kwotę przypisaną do tego progu z tabeli.
    6. LOGIKA BRAKU DANYCH (Koszty podróży/Erasmus): Jeśli odpowiedź o dofinansowaniu zależy od kilometrów, a użytkownik ich nie podał, nie odrzucaj pytania. Poproś go o podanie odległości lub miasta docelowego.
    7. TARCZA RODO: Jeśli użytkownik poda w czacie swoje wrażliwe dane osobowe (np. PESEL, numer albumu, nazwisko, adres), natychmiast przerwij odpowiedź i napisz: "Ze względów bezpieczeństwa proszę, nie podawaj tutaj swoich danych osobowych, takich jak PESEL czy numer albumu. Ten czat służy tylko do ogólnych pytań o regulamin. Twoje dane nie zostały nigdzie zapisane." Dopiero po tym ostrzeżeniu możesz spróbować odpowiedzieć na ewentualne merytoryczne pytanie z jego wiadomości.

    KONTEKST:
    {context}
    """

    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-6:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.chat.completions.create(model="gpt-4o-mini", messages=api_messages)
    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})