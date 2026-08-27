from langchain_core.prompts import PromptTemplate, prompt
from langchain_ollama import ChatOllama
from urllib3 import response


prompt = PromptTemplate.from_template(f"""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I dont know"
Context: {context}
Question: {question}
Answer: """)

model = ChatOllama(model="gemma4:e2b ")
chain = prompt | model
response = chain.invoke({
    "context": f""" The most recent advancements in NLP are being driven by Large Language Models(LLMs). 
    These models outperform their smaller counterparts and hae become invaluable for developers who are creating 
    application with NLP capabilities. Developers can tap into these models through Hugging Face's transformers' library, or by utilizing
    OpenAI and Cohere's offerings through the 'openai' and 'cohere' libraries, respectively.""", 
    "question": "What model providers offer LLMs?"
})

print(response.content)