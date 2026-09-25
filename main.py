from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from tools import TOOLS
import uuid

# global variables for easy config
MODEL = "gemini-3.5-flash-lite"
TEMP = None
SEED = None
SYSTEM_PROMPT = """
You are a helpful assistant.
"""

# create model from global config
def get_model():
    return ChatGoogleGenerativeAI(model=MODEL, temperature=TEMP, seed=SEED).bind(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True #disable AFC to get rid of warning
        )
    )

# prompt the agent
def get_response(prompt: str, agent, thread_config : dict) -> str:
    return agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config=thread_config
    )

# print just the message content using rich.markdown and rich.console
def print_pretty_response(console : Console, response : dict):
    pretty_response = "Response: " + response["messages"][-1].text
    console.print(Markdown(pretty_response))

# print complete model response
def print_detailed_response(response : str):
    for message in response["messages"]:
        print(message)
        print()


def main():
    # setup
    load_dotenv()
    console = Console()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    agent = create_agent(
    model=get_model(),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer = InMemorySaver()
    )

    # main chat loop
    while True:
        try:
            prompt = input("Input: ")
            if prompt.strip().lower() == "exit": break
            response = get_response(prompt, agent, thread_config)
        except EOFError:
            break

        # use either print func here
        print_pretty_response(console, response)
        #print_detailed_response(response)


if __name__ == "__main__":
    main()