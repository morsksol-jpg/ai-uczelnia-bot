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
    **System Status:** Ready (v4.2)
    
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
@st.cache_resource(show_spinner="Inicjalizacja modułu wiedzy SAM...")
def load_and_prepare_db(_api_key):
    embeddings = OpenAIEmbeddings(api_key=_api_key, model="text-embedding-3-small")
    persist_directory = "vector_db_v4_2"
    collection_name = "sam_knowledge_v4_2"
    
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
if prompt := st.chat_input("Zadaj pytanie systemowi SAM..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Wyszukiwanie w bazie (RAG)
    results = db.max_marginal_relevance_search(
        prompt, k=15, fetch_k=30, filter={"unit": wybrana_uczelnia.lower()}
    )
    
    context_parts = []
    for r in results:
        plik = os.path.basename(r.metadata.get("source", "Document"))
        strona = r.metadata.get("page", 0) + 1 
        context_parts.append(f"[FILE: {plik}, PAGE: {strona}]\n{r.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # --- SYSTEM PROMPT V4.2 (Krzysztof Adamiak Edition) ---
    system_prompt = f"""
    You are SAM (Smart Assistance Module), a sophisticated AI system for organizational knowledge created by Krzysztof Adamiak.
    
    STRICT LANGUAGE RULE:
    - You MUST identify the language of the user's question and respond EXCLUSIVELY in that language.
    - Even if the provided CONTEXT is in Polish, translate it accurately to the user's language.

    CORE RULES:
    1. GDPR SHIELD: If user provides personal data (name, ID, PESEL), stop and warn them in their language.
    2. FIDELITY & NUMERICAL RIGOR: Do not generalize. Be extremely precise with numbers (dates, student counts, amounts). If the text says "three", you MUST NOT say "five".
    3. PROACTIVE CLARIFICATION: If the query is broad but the rule is specific, you MUST point this out and ask for clarification in the user's language.
    4. CITATIONS: At the end of every answer, append: "[Source: filename.pdf, Page: X]". Always translate "Source" and "Page" to the user's language.
    5. DATA LIMIT: If information is missing, refer to the official contact point for {wybrana_uczelnia.upper()}.

    CONTEXT:
    {context}
    """

    api_messages = [{"role": "system", "content": system_prompt}]
    # Przesyłamy historię czatu (ostatnie 6 wiadomości)
    for msg in st.session_state.messages[-6:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Wywołanie modelu z zerową temperaturą (Zero Creativity = High Fact Fidelity)
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=api_messages,
        temperature=0
    )
    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})