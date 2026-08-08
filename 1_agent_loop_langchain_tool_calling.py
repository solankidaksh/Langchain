from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langchain_tavily import TavilySearch



MAX_ITERATIONS = 10
MODEL = "qwen3.5:72b"

@tool
def get_product_price(product :str) -> float:
    """Look up for the price of a product in a catalog"""
    print(f" >>Executing get_product_rpice(product='{product}')")
    price = {'laptop':1000, 'mouse': 500, 'keyboard': 700}
    return price.get(product, 0)

@tool
def apply_discount(price :float, discount_tier :str) -> float:
    """Apply a discount tier to a price and return the final price
    Available discount tiers: bronze, gold, silver"""
    print(f" >>Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {'bronze': 5, 'gold':10, 'silver':7}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

@traceable(name="Langchain Agent Loop")
def run_agent(question :str):
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name : t for t in tools}
    llm = init_chat_model(f"Ollama:{MODEL}", temperature = 0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content = (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ), 
        HumanMessage(content = question)
    ] 


if __name__ == "__main__":
    print("Hello Langchain Agent (.binds_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")
