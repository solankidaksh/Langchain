from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_TERATIONS = 10
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
    pass

if __name__ == "__main__":
    print("Hello Langchain Agent (.binds_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")
