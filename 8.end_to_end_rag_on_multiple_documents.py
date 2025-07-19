# pip install unstructured
# pip install docx
# pip install python-docx pypdf

import os
from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import data_info
# --------- Config ---------
DATA_DIR = "data"  # Directory containing your files
CHROMA_PERSIST_DIR = "chroma_store_multiple_document_openai"


LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"
OPENAI_API_KEY = data_info.open_ai_key # Replace this with your open ai key "SK-"

# --------- Helper function to load documents ---------
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        # elif filename.endswith(".docx"):
        #     loader = UnstructuredWordDocumentLoader(file_path)
        else:
            print(f"[WARN] Skipping unsupported file type: {filename}")
            continue

        docs = loader.load()
        documents.extend(docs)
    return documents

# --------- Load and Split Documents ---------
print("[INFO] Loading and splitting documents...")
raw_docs = load_documents_from_directory(DATA_DIR)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# --------- Initialize Embeddings ---------
EMBED_MODEL = "text-embedding-3-small"
print(f"[INFO] Using OpenAI Embeddings: {EMBED_MODEL}")
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL,openai_api_key=OPENAI_API_KEY)

# --------- Create / Load Vector Store ---------
if os.path.exists(CHROMA_PERSIST_DIR):
    print("[INFO] Loading existing Chroma vector store...")
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
else:
    print("[INFO] Creating Chroma vector store and embedding documents...")
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PERSIST_DIR)
    vectordb.persist()

# --------- Load LLM ---------
# --------- Initialize LLM ---------
print(f"[INFO] Using OpenAI LLM: {LLM_MODEL}")
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0,openai_api_key=OPENAI_API_KEY)

# --------- Setup RetrievalQA ---------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce"
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# --------- Q&A Loop ---------
print("\n[READY] Ask me anything about the content of your files. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    query = "Using only the given context answer following question:\n" + query
    result = qa_chain(query)
    print(result)
    print("\nAI:", result["result"], "\n")
