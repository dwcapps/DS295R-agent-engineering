import asyncio
import uuid
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEMORY_FILE = PROJECT_ROOT / "data" / "memory.jsonl"
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

import logging
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from dotenv import load_dotenv
from google.genai import types
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown

from ds295r_agent_engineering.tools import TOOL_PACKAGE
from ds295r_agent_engineering.shopping_agent import talk_to_shopping_agent


# Configuration
MODEL = "gemini-flash-lite-latest"
TEMPERATURE = None
SEED = None
ENABLE_CHECKPOINTING = True


SYSTEM_PROMPT =  """
You are a multipurpose AI agent.

Use the LangChain documentation tools when answering questions about LangChain.

You also have access to persistent memory tools.

You can communicate with a Shopping subagent to help the user find good deals.

Memory guidelines:
- Search your memory for relevant information before answering personalized questions.
- Save stable facts, preferences, goals, and important project details when appropriate.
- Do not save every ordinary conversational detail.
- Do not save sensitive personal information unless the user explicitly asks you to remember it.
- When the user explicitly says "remember this," save the information using the memory tools.
- Never claim to have saved something unless the memory tool succeeds.
"""

MCP_CONFIG = {
    "langchain-docs": {
        "url": "https://docs.langchain.com/mcp"
    },

    "memory": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-memory"
      ],
      "env": {
        "MEMORY_FILE_PATH": str(MEMORY_FILE)
      }
    }
}


def create_model() -> ChatGoogleGenerativeAI:
    """Create and configure the Gemini model."""
    model = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
    )
    # Disable Google's automatic function calling because
    # LangChain is managing tool calls itself.
    return model.bind(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )


def create_thread_config() -> dict[str, dict[str, str]]:
    """Create a unique conversation/thread configuration."""
    return {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }


async def get_response(
    prompt: str,
    agent: Any,
    thread_config: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Send a prompt to the agent and return its response."""
    return await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        config=thread_config,
    )


def print_response(console: Console, response: dict[str, Any]) -> None:
    """Print only the agent's final response as formatted Markdown."""
    message = response["messages"][-1]
    console.print(Markdown(f"Response:\n\n{message.text}"))


def print_detailed_response(response: dict[str, Any]) -> None:
    """Print every message produced during the agent's execution."""
    for message in response["messages"]:
        print(message)
        print()


async def chat_loop(
    agent: Any,
    console: Console,
    thread_config: dict[str, dict[str, str]],
) -> None:
    """Run the interactive command-line chat loop."""
    while True:
        console.rule()

        try:
            prompt = input("Input: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() == "exit":
            break
        if not prompt.strip():
            continue
        response = await get_response(
            prompt,
            agent,
            thread_config,
        )
        print_response(console, response)
        # print_detailed_response(response)


async def main() -> None:
    load_dotenv()

    console = Console()
    thread_config = create_thread_config()

    adapter = MCPAdapter(MCP_CONFIG)
    mcp_tools = await adapter.list_tools()

    print("Loaded MCP tools:")
    for tool in mcp_tools:
        print(f"  - {tool.name}")

    all_tools = TOOL_PACKAGE + mcp_tools

    agent = create_agent(
        model=create_model(),
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=(
            InMemorySaver()
            if ENABLE_CHECKPOINTING
            else None
        ),
    )

    await chat_loop(
        agent,
        console,
        thread_config,
    )


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()