# connect_memory_with_llm.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

import os

# ========================
# CONFIG
# ========================
HF_TOKEN = "HF_TOKEN"  # Replace with your HF token
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
VECTORSTORE_PATH = r"C:\Users\admin\Desktop\Hugging Face Projects\Medical-Chatbot\vectorstore\db_faiss"
DATA_PATH = r"C:\Users\admin\Desktop\Hugging Face Projects\Medical-Chatbot\data\The_GALE_ENCYCLOPEDIA_OF_MEDICINE_Volume_1.pdf"

# ========================
# 1. LOAD EMBEDDINGS
# ========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ========================
# 2. LOAD OR CREATE VECTORSTORE
# ========================
if os.path.exists(VECTORSTORE_PATH):
    print("[INFO] Loading existing FAISS vectorstore...")
    vectorstore = FAISS.load_local(
        folder_path=VECTORSTORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Needed in recent LangChain versions
    )
else:
    print("[INFO] Creating new vectorstore from documents...")
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

# ========================
# 3. LOAD LLM (Mistral conversational mode)
# ========================
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="conversational",   # ✅ FIX — matches supported task
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

llm = load_llm(MODEL_ID)

# ========================
# 4. CREATE QA CHAIN
# ========================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True,
    input_key="query"  # ✅ matches invoke({"query": ...})
)

# ========================
# 5. INTERACTIVE LOOP
# ========================
print("\n=== Interactive QA ===")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("Ask: ")
    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = qa_chain.invoke({"query": user_query})

    print("\n[Answer]:", response["result"])
    print("\n[Sources]:")
    for doc in response["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"))
    print("\n" + "="*50 + "\n")
