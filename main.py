from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import uuid
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import TOOLS

MODEL = "gemini-3.5-flash-lite"
TEMP = None
SEED = None
SYSTEM_PROMPT = """
You are a helpful assistant.
"""

def get_response(prompt: str, agent, thread_config : dict) -> str:
    return agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config=thread_config
    )

def get_model():
    return ChatGoogleGenerativeAI(model=MODEL, temperature=TEMP, seed=SEED).bind(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )

def print_pretty_response(console : Console, response : dict):
    pretty_response = "Response: " + response["messages"][-1].text
    console.print(Markdown(pretty_response))

def print_detailed_response(response : str):
    for message in response["messages"]:
        print(message)
        print()

def main():
    load_dotenv()
    console = Console()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    agent = create_agent(
    model=get_model(),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer = InMemorySaver()
    )

    while True:
        try:
            prompt = input("Input: ")
            if prompt.strip().lower() == "exit": break
            response = get_response(prompt, agent, thread_config)
        except EOFError:
            break
        print_pretty_response(console, response)
        #print_detailed_response(response)

if __name__ == "__main__":
    main()