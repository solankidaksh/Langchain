import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_community.document_loaders import TextLoader
from langchain_postgres import PGVectorStore, PGEngine
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
engine = PGEngine.from_connection_string(url=connection)

table_name = "my_docs"

raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
documents = text_splitter.split_documents(raw_documents)

model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

db = PGVectorStore.from_documents(
    documents, model, engine=engine, table_name=table_name)
results = db.similarity_search("what is captial of India?", k=4)
print(results)

print("Adding documents to the vectore store databse")
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents([
    Document(
        page_content = "New Delhi is the capital of India",
        metadata = {"location": "New Delhi", "country": "India"}
    )
],ids=ids)
print("Documents added. Document counts:", len(db.get_by_ids(ids)))