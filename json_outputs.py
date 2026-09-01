from pydantic import BaseModel
from langchain_ollama import ChatOllama

class AnswerWithJustification(BaseModel):
    '''Answer the question with the justification of the answer'''

    answer: str
    '''The answer to the user's question'''
    justification: str
    '''The justification for the answer'''
llm = ChatOllama(model = 'gemma4:e2b')
structured_llm = llm.with_structured_output(AnswerWithJustification)
response = structured_llm.invoke('''What weighs more a pound of feathers or a pound of gold?''')
print(response)