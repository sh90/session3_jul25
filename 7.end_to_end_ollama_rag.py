#pip install langchain
#pip install chromadb
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# --------- Config ---------
TEXT_FILE_PATH = "data/onboarding.txt"  # Your text file path
CHROMA_PERSIST_DIR = "chroma_store"

# --------- Load Text ---------
print("[INFO] Loading and splitting text...")
with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

raw_docs = [Document(page_content=text)]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
docs = splitter.split_documents(raw_docs)

# --------- Initialize Embedding ---------
print("[INFO] Generating embeddings using mxbai-embed-large...")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")



# --------- Create / Load Vector Store ---------
if os.path.exists(CHROMA_PERSIST_DIR):
    print("[INFO] Loading existing Chroma vector store...")
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
else:
    print("[INFO] Creating Chroma vector store and embedding documents...")
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PERSIST_DIR)
    vectordb.persist()


# --------- Load LLM ---------
print("[INFO] Loading gemma:1b model via Ollama...")
llm = Ollama(model="gemma3:1b")

# --------- Setup RetrievalQA ---------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce" for more complex tasks
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# --------- Q&A Loop ---------
print("\n[READY] Ask me anything about the content in your file. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain(query)
    print("\nAI:", result["result"], "\n")
