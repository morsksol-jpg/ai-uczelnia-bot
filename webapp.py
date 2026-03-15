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
st.set_page_config(page_title="SAM - Smart Assistance Module", page_icon="🛡️", layout="wide")

# Ukrycie standardowych elementów Streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- NOWY BRANDING (SAM) ---
st.markdown("<h1 style='text-align: center; font-size: 60px;'>SAM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; font-weight: bold;'>Smart Assistance Module</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px; color: gray;'>AI system for organizational knowledge</p>", unsafe_allow_html=True)
st.markdown("---")

# --- PANEL BOCZNY ---
with st.sidebar:
    st.markdown("### ⚙️ Konfiguracja Systemu")
    lista_uczelni = ["merito", "uw", "uj"]
    wybrana_uczelnia = st.selectbox("Wybierz jednostkę organizacyjną:", lista_uczelni)
    st.markdown("---")
    st.info("Status: System Ready (v4.0)\n\nFidelity Mode: Active\nGDPR Shield: Active\n**Multilingual:** Enabled")
    st.markdown("---")
    st.caption("Copyright (c) 2026 Krzysztof Adamiak. All rights reserved.")

st.title(f"🎓 SAM – Studencki Asystent ({wybrana_uczelnia.upper()})")

# Pobranie klucza API
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Brak klucza API OpenAI!")
    st.stop()

# --- FUNKCJA ŁADOWANIA I OPTYMALIZACJI BAZY (V3) ---
@st.cache_resource(show_spinner="Inicjalizacja szybkiej bazy wiedzy v3...")
def load_and_prepare_db(_api_key):
    embeddings = OpenAIEmbeddings(api_key=_api_key, model="text-embedding-3-small")
    
    # ZMIANA NA V3: Zmuszamy chmurę do zbudowania bazy od nowa
    persist_directory = "vector_db_v3"
    collection_name_v3 = "sam_production_v3"
    
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings, 
            collection_name=collection_name_v3
        )

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
                            doc.metadata["uczelnia"] = uczelnia_folder.lower()
                        documents.extend(loaded_docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        collection_name=collection_name_v3, 
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Zadaj pytanie dotyczące regulaminu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    results = db.max_marginal_relevance_search(
        prompt, 
        k=25, 
        fetch_k=50,
        filter={"uczelnia": wybrana_uczelnia.lower()}
    )
    
    # ZMIANA: Czyste tagi maszynowe do cytowań
    context_parts = []
    for r in results:
        plik = os.path.basename(r.metadata.get("source", "Document"))
        strona = r.metadata.get("page", 0) + 1 
        tresc = r.page_content
        context_parts.append(f"[FILE: {plik}, PAGE: {strona}]\n{tresc}")
        
    context = "\n\n---\n\n".join(context_parts)

    kontakty_dziekanatow = {
        "merito": "Dziekanat Merito Szczecin: ul. Śniadeckich 3, Tel: +48 91 422 74 44, E-mail: dziekanat@szczecin.merito.pl",
        "uw": "Dziekanat UW: ul. Krakowskie Przedmieście 26/28, 00-927 Warszawa",
        "uj": "Dziekanat UJ: ul. Gołębia 24, 31-007 Kraków"
    }
    
    obecny_kontakt = kontakty_dziekanatow.get(wybrana_uczelnia.lower(), "Contact your university's main office.")

    # ZMIANA: Twardy anglojęzyczny prompt kontrolujący model AI
    system_prompt = f"""
    You are a professional student assistant for {wybrana_uczelnia.upper()} university.
    CRITICAL DIRECTIVE: You MUST answer in the EXACT SAME LANGUAGE that the user used in their prompt. If the user asks in English, reply in English. If German, reply in German. If Polish, reply in Polish. NO EXCEPTIONS.

    RULES:
    1. GDPR PRIVACY SHIELD (PRIORITY 1): If the user shares sensitive data (PESEL, student ID, Matrikelnummer, numer albumu, name, address), IMMEDIATELY stop and reply: "For security reasons, please do not share personal data here. Your data has not been saved." (TRANSLATE this warning into the user's language). Do not answer their question.
    2. If the user asks how many people are required to establish a student organization, use only numbers explicitly referring to the founders list ("lista założycieli") of the organization.
    3. MISSING DATA: If the answer is not in the provided text, DO NOT make it up. Reply: "I'm sorry, I couldn't find this information. Please contact the Dean's office: {obecny_kontakt}" (TRANSLATE this phrase into the user's language).
    4. SCHOLARSHIP MATH: Scholarships depend on TOTAL POINTS (GPA + extra points). If the user provides both, ADD them mathematically (e.g., 4.8 + 2.0 = 6.8). Find the matching range in the context table (e.g., 6.5 - 6.99) and provide the EXACT monetary amount.
    5. PRECISION AND FIDELITY (CRITICAL): Pay strict attention to adjectives and specific conditions in the text. If a rule applies to a specific type of event (e.g., "egzamin dyplomowy" / "diploma exam"), you MUST NOT generalize it to all exams. You must explicitly state the specific condition in your answer. If the user's question is broad, but the context is specific, clarify this limitation.
    6. PROACTIVE CLARIFICATION: If the user's query is broad (e.g., "I failed an exam"), but the retrieved context only covers a highly specific scenario (e.g., "diploma exam"), you must point out this difference. You must state what the specific rule says, but immediately explicitly ASK the user to clarify their situation (e.g., "I found rules regarding a diploma exam. Is this what you mean, or are you asking about a regular session exam?"). Always translate this clarifying question into the user's language.
    7. CITATIONS: At the very end of your response, you MUST append the source file and page number. 
    Format strictly as: "[Source: filename.pdf, Page: X]".
    CRITICAL WARNING: You MUST translate the words "Source" and "Page" into the EXACT language of the user's prompt! NEVER use Polish words like "Źródło" or "Strona" unless the user asked the question in Polish! If the user asks in English, you MUST use "[Source: ..., Page: ...]".
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