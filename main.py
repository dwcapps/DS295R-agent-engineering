from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import uuid
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-3.5-flash-lite"
SYSTEM_PROMPT = \
"""
You are a helpful assistant.
"""

def get_response(prompt: str, agent, thread_config : dict) -> str:
    response = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    thread_config
    )
    return "Response: " + response["messages"][-1].text

def get_model():
    model = ChatGoogleGenerativeAI(model=MODEL) \
        .bind(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    return model

def main():
    load_dotenv()
    console = Console()
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    checkpointer = InMemorySaver()
    )

    while True:
        try:
            prompt = input("Input: ")
            if prompt == "exit": break
            response = get_response(prompt, agent, thread_config)
        except EOFError:
            break
        console.print(Markdown(response))

if __name__ == "__main__":
    main()