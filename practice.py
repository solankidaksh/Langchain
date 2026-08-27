from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


template = PromptTemplate.from_template(f"""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I dont know"
Context: {context}
Question: {question}
Answer: """)

template.invoke({
    "context": f""" The most recent advancements in NLP are being driven by Large Language Models(LLMs). 
    These models outperform their smaller counterparts and hae become invaluable for developers who are creating 
    application with NLP capabilities. Developers can tap into these models through Hugging Face's transformers' library, or by utilizing
    OpenAI and Cohere's offerings through the 'openai' and 'cohere' libraries, respectively.""", 
    "question": "What model providers offer LLMs?"
})