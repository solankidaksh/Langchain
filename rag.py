import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVectorStore, PGEngine

loader = TextLoader("./test.txt")
doc = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(doc)

embedding_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
engine = PGEngine.from_connection_string(url=connection)

table_name = "my_docs"

# it is only used once to create the table, if run twice then it will throw an error. 
# engine.init_vectorstore_table(
#     table_name = table_name, 
#     vector_size = 384
# )
db = PGVectorStore.from_documents(chunks, embedding_model, engine=engine, table_name=table_name)
embeddings = embedding_model.embed_documents(
    chunk.page_content for chunk in chunks
)
print(embeddings)