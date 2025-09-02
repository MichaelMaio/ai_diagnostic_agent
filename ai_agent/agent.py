from groq import Groq
import os
import re
import requests
from typing import Any, Dict, List, Optional


# ReAct-based agent — keeps conversation history and calls the model hosted on Groq.
class ReActAgent:

    def __init__(self) -> None:   
        
        # Initialize the Groq client.
        self.groq_client: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Load the system prompt from a file.
        system_prompt: str = open("ai_agent/system_prompt.txt").read().strip()

        # Conversation history (system + user + assistant messages).
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def step(self, user_input: Optional[str] = None) -> str:

        # Executes one step of the agent loop and returns the assistant's response.
        if user_input and user_input.strip():
            self.messages.append({"role": "user", "content": user_input})

        # Call the Groq chat completion API.
        completion = self.groq_client.chat.completions.create(
            messages=self.messages,
            model="llama-3.3-70b-versatile"
        )

        # Extract the assistant's reply.
        content: str = completion.choices[0].message.content

        # Add the reply to the conversation history.
        self.messages.append({"role": "assistant", "content": content})
        return content


# Executes a tool call by sending it to the MCP server.
def do_action(result: str) -> str:

    # Match: Action: FunctionName[: optional arguments]
    match = re.search(r"Action:\s*([a-zA-Z0-9_]+)(?::\s*(.+))?", result)

    if not match:
        return None, None

    tool = match.group(1).strip()
    raw_args = match.group(2)

    if raw_args:
        # Split by comma and strip quotes and whitespace
        args = [arg.strip().strip("'\"") for arg in raw_args.split(",")]
    else:
        args = []

    try:
        # Prepare the MCP server JSON-RPC payload.
        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": tool,
            "params": {"input": args},
            "id": "1"
        }

        # Call the MCP server.
        resp = requests.post("http://localhost:5000/mcp", json=payload, timeout=10)
        resp.raise_for_status()
        mcp_result: Dict[str, Any] = resp.json()

        # Extract the result or error message.
        if "result" in mcp_result:
            observation: str = mcp_result["result"]
        elif "error" in mcp_result:
            observation = f"Error: {mcp_result['error'].get('message', 'Unknown error')}"
        else:
            observation = "Error: Invalid MCP server response"

    except Exception as e:
        observation = f"Error calling MCP server: {e}"

    next_prompt: str = f"Observation:\n{observation}"
    print(next_prompt)
    return next_prompt


# Main agent loop — runs up to max_iterations.
def run_agent() -> None:

    # Initialize services.
    agent = ReActAgent()

    next_prompt: str = "Question: My test failed when run against my app. What is the fix for the failure?" 
    max_iterations: int = 10

    # Run the agent loop.
    for _ in range(max_iterations):
        result: str = agent.step(next_prompt)
        print(result)

        # Check for Action or Answer in the result.
        if "PAUSE" in result and "Action" in result:
            next_prompt = do_action(result)
            continue

        answer_prefix: str = "Answer: "

        if answer_prefix in result:
            break


if __name__ == "__main__":
    run_agent()