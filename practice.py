from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable
from langchain.messages import HUmanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool

max_iteration = 10
model = gemma4:e2b

@tool
def product_price(product :str) -> float:
    price = {'laptop':1000, 'mouse':500, 'keyboard':700}
    return price.get(product, 0)

@tool 
def discount(price :float, discount_tier :str) -> float:
    discount_percentage = {'bronze':5, 'gold': 10, 'silver':7}
    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)
 
@traceable(name="Langchain Agent Loop")
def run_agent(question :str):
    tools = [product_price, discount]
    tool_dict = { t.name : t for t in tools}
    llm = init_chat_model(model, model_provider = 'ollama', temperature = 0)
    llm_with_tools = llm.bind_tools(tools)
    print(question)

    for iteration in range(1, max_iteration+1):
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args')
        tool_id = tool_call.get('id')

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError("No tool found")
        observation = tool_to_use.invoke(tool_args)
        print(observation)

        message.append(ai_message)
        message.append(ToolMessage(content=str(observation), tool_call_id = tool_id))
        
        