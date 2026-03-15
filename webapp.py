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
    st.info("""
    **System Status:** Ready (v4.0)
    
    🛡️ **Fidelity Mode:** Active
    🔒 **GDPR Shield:** Active
    🌍 **Multilingual:** Enabled
    """)
    st.markdown("---")
    st.caption("Copyright (c) 2026 Krzysztof Adamiak. All rights reserved.")

# Pobranie klucza API
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Brak klucza API OpenAI!")
    st.stop()

# --- INICJALIZACJA BAZY WIEDZY ---
@st.cache_resource(show_spinner="Inicjalizacja modułu wiedzy...")
def load_and_prepare_db(_api_key):
    embeddings = OpenAIEmbeddings(api_key=_api_key, model="text-embedding-3-small")
    persist_directory = "vector_db_v4"
    collection_name = "sam_knowledge_v4"
    
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)

    documents = []
    base_folder = "documents"
    if os.path.exists(base_folder):
        for unit_folder in os.listdir(base_folder):
            unit_path = os.path.join(base_folder, unit_folder)
            if os.path.isdir(unit_path):
                for filename in os.listdir(unit_path):
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(os.path.join(unit_path, filename))
                        loaded_docs = loader.load()
                        for doc in loaded_docs:
                            doc.metadata["unit"] = unit_folder.lower()
                        documents.extend(loaded_docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings, collection_name=collection_name, persist_directory=persist_directory)
    return db

db = load_and_prepare_db(api_key)
client = OpenAI(api_key=api_key)

# Pamięć sesji
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_unit" not in st.session_state:
    st.session_state.last_unit = wybrana_uczelnia

if st.session_state.last_unit != wybrana_uczelnia:
    st.session_state.messages = []
    st.session_state.last_unit = wybrana_uczelnia
    st.rerun()

# Wyświetlanie czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- GŁÓWNA LOGIKA SYSTEMU ---
if prompt := st.chat_input("Zadaj pytanie systemowi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    results = db.max_marginal_relevance_search(
        prompt, k=15, fetch_k=30, filter={"unit": wybrana_uczelnia.lower()}
    )
    
    context_parts = []
    for r in results:
        plik = os.path.basename(r.metadata.get("source", "Document"))
        strona = r.metadata.get("page", 0) + 1 
        context_parts.append(f"[FILE: {plik}, PAGE: {strona}]\n{r.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # --- SYSTEM PROMPT V4 (SOPHISTICATED & SECURE) ---
    system_prompt = f"""
    You are SAM (Smart Assistance Module), a sophisticated AI system for organizational knowledge.
    
    CRITICAL DIRECTIVE: You MUST answer in the EXACT SAME LANGUAGE as the user. NO EXCEPTIONS.

    CORE RULES:
    1. GDPR SHIELD: If user provides personal data (name, ID, PESEL, Matrikelnummer), stop immediately and warn the user. Translate warning to their language.
    2. FIDELITY & PRECISION: Never generalize specific terms. If a rule mentions "diploma exam", do not call it just an "exam". 
    3. PROACTIVE CLARIFICATION: If the user's query is vague, but the context is specific, you MUST ask for clarification (e.g., "I found rules for diploma exams. Are you asking about those or regular ones?").
    4. CITATIONS: At the end of every answer, append the source as: "[Source: filename.pdf, Page: X]". Translate "Source" and "Page" to the user's language.
    5. DATA LIMIT: If information is missing, refer to the official contact point for {wybrana_uczelnia.upper()}.

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