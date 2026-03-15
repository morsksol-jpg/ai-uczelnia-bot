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
    
    # Budowanie kontekstu z nazwami plików i numerami stron
    context_parts = []
    for r in results:
        # Wyciągamy samą nazwę pliku z długiej ścieżki (np. z "documents/merito/regulamin.pdf" robimy "regulamin.pdf")
        plik = os.path.basename(r.metadata.get("source", "Regulamin"))
        # PyPDFLoader liczy strony od 0, więc dodajemy 1, żeby było naturalnie
        strona = r.metadata.get("page", 0) + 1 
        tresc = r.page_content
        context_parts.append(f"[ŹRÓDŁO - PLIK: {plik}, STRONA: {strona}]\n{tresc}")
        
    context = "\n\n---\n\n".join(context_parts)

    # Słownik z kontaktami "na sztywno" dla każdej uczelni
    kontakty_dziekanatow = {
        "merito": "Dziekanat Merito Szczecin: ul. Śniadeckich 3, Tel: +48 91 422 74 44, E-mail: dziekanat@szczecin.merito.pl",
        "uw": "Dziekanat UW: ul. Krakowskie Przedmieście 26/28, 00-927 Warszawa",
        "uj": "Dziekanat UJ: ul. Gołębia 24, 31-007 Kraków"
    }
    
    # Wyciągamy kontakt dla aktualnie wybranej uczelni
    obecny_kontakt = kontakty_dziekanatow.get(wybrana_uczelnia.lower(), "Skontaktuj się z głównym dziekanatem swojej uczelni.")

    system_prompt = f"""
    You are a professional student assistant for {wybrana_uczelnia.upper()} university.
    CRITICAL DIRECTIVE: You MUST answer in the EXACT SAME LANGUAGE that the user used in their prompt. If the user asks in English, reply in English. If German, reply in German. If Ukrainian, reply in Ukrainian. NO EXCEPTIONS.

    RULES:
    1. GDPR PRIVACY SHIELD (PRIORITY 1): If the user shares sensitive data (PESEL, student ID, Matrikelnummer, name, address), IMMEDIATELY stop and reply: "For security reasons, please do not share personal data here. Your data has not been saved." (TRANSLATE this warning into the user's language). Do not answer their question.
    2. MISSING DATA: If the answer is not in the provided text, DO NOT make it up. Reply: "I'm sorry, I couldn't find this information. Please contact the Dean's office: {obecny_kontakt}" (TRANSLATE this phrase into the user's language).
    3. SCHOLARSHIP MATH: Scholarships depend on TOTAL POINTS (GPA + extra points). If the user provides both, ADD them mathematically (e.g., 4.8 + 2.0 = 6.8). Find the matching range in the context table (e.g., 6.5 - 6.99) and provide the EXACT monetary amount.
    4. CITATIONS: Always append the source at the end of your answer, formatted in the user's language (e.g., in English: [Source: file.pdf, Page: 5], in Spanish: [Fuente: archivo.pdf, Página: 5]).

    CONTEXT:
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