from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import uuid
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
console = Console()
thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}


def get_response(prompt: str, agent) -> str:
    response = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    thread_config
    )
    return response["messages"][-1].text

def main():
    
    agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[],
    system_prompt="You are a helpful assistant",
    checkpointer = InMemorySaver()
    )

    while True:
        try:
            prompt = input("Input: ")
            if prompt == "exit": break
            response = get_response(prompt, agent)
        except EOFError:
            break
        console.print(Markdown(response))

if __name__ == "__main__":
    main()