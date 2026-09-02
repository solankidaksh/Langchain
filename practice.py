from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from urllib3 import response
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./practice.pdf")
pages = loader.load()

prompt = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I dont know'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])

model = ChatOllama(model="gemma4:e2b")
chain = prompt | model
response = chain.invoke({
    "context": """ The most recent advancements in NLP are being driven by Large Language Models(LLMs). 
    These models outperform their smaller counterparts and hae become invaluable for developers who are creating 
    application with NLP capabilities. Developers can tap into these models through Hugging Face's transformers' library, or by utilizing
    OpenAI and Cohere's offerings through the 'openai' and 'cohere' libraries, respectively.""", 
    "question": "What model providers offer LLMs?"
})

print(response.content)