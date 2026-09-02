from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./practice.pdf")
pages = loader.load()