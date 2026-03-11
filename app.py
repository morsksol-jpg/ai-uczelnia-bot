import os
from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
documents = []

folder = "documents"

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder, filename))
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

db_path = "vector_db"

if os.path.exists(db_path):
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=db_path
    )
    db.persist()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
while True:
    question = input("\nStudent: ")

    if question.lower() in ["exit", "quit", "koniec"]:
        print("Bot: Do zobaczenia!")
        break

    results = db.similarity_search(question, k=6)

    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
Na podstawie poniższych fragmentów regulaminów uczelni odpowiedz na pytanie studenta.

Fragmenty regulaminu:
{context}

Pytanie studenta:
{question}

Odpowiedź:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nBot:\n")
    print(response.choices[0].message.content)