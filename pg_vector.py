from langchain_community.document_loaders import TextLoader
from langchain_postgres.vectorstores import PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

from rag import connection

connnection = 'postgressql+psycopg://langchain:langchain@localhost:6024/langchain'

raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
documents = text_splitter.split_documents(raw_documents)

model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

db = PGVectorStore.from_documents(
    documents, model, connection=connection)
db.similarity_search("what is captial of India?", k=4)
