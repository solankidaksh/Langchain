from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVectorStore
from langchain_core.documents import Document
import uuid

loader = TextLoader("./test.txt")
doc = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(doc)

embedding_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
connection = 'postgressql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVectorStore.from_documents(chunks, embedding_model, connection=connection)
embeddings = embedding_model.embed_documents(
    chunk.page_content for chunk in chunks
)
print(embeddings)