from langchain.tools import tool
from langchain_core.tools import BaseTool

from datetime import datetime

@tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %I:%M:%S %p")


# this tool tests that the agent is actually using tools and not relying on existing 
#   knowledge by providing a hardcoded value that the agent could not have known beforehand.
@tool
def get_secret_passphrase() -> str:
    """Return the secret passphrase."""
    return "the black moon howls"


# dynamically add all @tool functions in this file to TOOLS list for easy exporting.
TOOLS = [
    obj
    for name, obj in globals().items()
    if isinstance(obj, BaseTool)
    and not name.startswith("_")
]