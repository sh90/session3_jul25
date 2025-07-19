from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "data/spgi-annual-report-2023.pdf"
loader = PyPDFLoader(file_path)
docs1 = loader.load()

print("Done")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs1)

print("Done")
