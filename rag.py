from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

loader = TextLoader("./test.txt")
doc = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 20
)
chunks = splitter.split_documents(doc)

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
embeddings = embedding_model.embed_documents(
    chunk.page_content for chunk in chunks
)
