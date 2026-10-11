from dotenv import load_dotenv
load_dotenv()

import uuid

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """
You are a shopping research specialist.

Your job is to find products that meet the user's requirements and compare
available listings.

Always:
- Pay attention to the user's budget and requirements.
- Distinguish facts from guesses.
- Compare important tradeoffs such as price, quality, shipping, and reviews.
- Never purchase anything.
- Never claim that a product is available unless a tool confirms it.
- Return a concise recommendation with alternatives.
"""

def find_product(product: str) -> str:
    """docstring"""
    pass

SHOPPING_AGENT = create_agent(
    model="google_genai:gemini-flash-lite-latest",
    tools=[find_product],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

thread_config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

def talk_to_shopping_agent(prompt: str) -> str:
    """Talk to the shopping agent and get a response. `prompt` is what you tell the agent."""
    response = SHOPPING_AGENT.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        thread_config
    )

    return response["messages"][-1].text