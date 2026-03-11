import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

st.title("🎓 Bot uczelni – przewodnik po biurokracji")

# 1. OPTYMALIZACJA: Baza ładuje się tylko RAZ, a nie przy każdym pytaniu!
@st.cache_resource
def load_and_prepare_db():
    embeddings = HuggingFaceEmbeddings()
    documents = []
    folder = "documents"

    # Zabezpieczenie: sprawdza czy folder istnieje
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder, filename))
                documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(texts, embeddings)
    return db

# Uruchomienie bazy z pamięci podręcznej
db = load_and_prepare_db()

# Klient OpenAI (Pamiętaj o wpisaniu klucza w Streamlit Secrets!)
# Streamlit Cloud używa st.secrets, lokalnie load_dotenv() łapie z .env
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# pamięć rozmowy
if "messages" not in st.session_state:
    st.session_state.messages = []

# wyświetlanie historii
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# pole pytania
if prompt := st.chat_input("Zadaj pytanie"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # wyszukiwanie fragmentów (bez nadmiernego słowa "warunki", żeby nie mylił AI)
    results = db.similarity_search(prompt + " regulamin stypendium", k=8)

    # 2. NAPRAWA: Usunięto agresywny filtr na "załącznik". 
    # AI dostanie czysty tekst i samo wybierze odpowiedź.
    context = "\n\n".join([r.page_content for r in results])

    # prompt dla AI
    full_prompt = f"""
    Odpowiadaj WYŁĄCZNIE na podstawie poniższych fragmentów regulaminu.
    Jesteś profesjonalnym i pomocnym dyspozytorem/asystentem studenta.
    Jeśli odpowiedź nie znajduje się we fragmentach, napisz dokładnie:
    "Przepraszam, ale nie znalazłem tej informacji w aktualnym regulaminie. Skontaktuj się z dziekanatem."

    Fragmenty regulaminu:
    {context}

    Pytanie studenta:
    {prompt}

    Odpowiedź:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})