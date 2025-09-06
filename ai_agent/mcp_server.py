# cd ai_agent
# python -m venv venv
# .\venv\Scripts\activate
# pip install -r requirements.txt
# python mcp_server.py

import os

import inspect

from flask import Flask, request, jsonify

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct, SearchRequest
from sentence_transformers import SentenceTransformer

# For a production environment, use a more robust server like FastAPI. But Flask is sufficient for this demo.
app = Flask(__name__)


# -------------------------------
# Tool Implementations
# -------------------------------


def get_relevant_code(query) -> str:

    model = SentenceTransformer("BAAI/bge-large-en")
    embedding = model.encode([query], normalize_embeddings=True)[0]

    client = QdrantClient(host="localhost", port=6333)

    results = client.search(
        collection_name="code_chunks",
        query_vector=embedding,
        limit=5
    )

    return "\n".join([f"{hit.payload['name']} ({hit.payload['filePath']})" for hit in results])

def get_code_file_contents(filename) -> str:

    search_root = os.path.abspath("../ecommerce-website")

    for root, dirs, files in os.walk(search_root):
        if filename in files:
            full_path = os.path.join(root, filename)
            print(f"Found: {full_path}")

    return open(full_path).read().strip()

def get_list_of_code_files() -> list[str]:

    root_path = os.path.abspath("../ecommerce-website")
    valid_exts = {".tsx", ".ts", ".html", ".css"}
    matching_files = []

    for dirpath, dirs, filenames in os.walk(root_path):
        for filename in filenames:
            if os.path.splitext(filename)[1] in valid_exts:
                matching_files.append(filename)

    return matching_files

def invoke_tool(tool_func, params):

    # Determine the expected parameters of the tool function.
    sig = inspect.signature(tool_func)
    param_names = list(sig.parameters.keys())

    # Extract the "input" field.
    input_data = params.get("input", None)

    # If no input is provided and the function takes no parameters, call it directly.
    if input_data is None:
        return tool_func()

    # If input is a list and matches expected arity, unpack it.
    if isinstance(input_data, list) and len(input_data) == len(param_names):
        return tool_func(*input_data)

    # If input is a single value and only one param is expected.
    if isinstance(input_data, str) and len(param_names) == 1:
        return tool_func(input_data)

    # If input is a dict and matches param names.
    if isinstance(input_data, dict):
        return tool_func(**input_data)

    raise ValueError(f"Cannot map input {input_data} to function {tool_func.__name__}")


# -------------------------------
# JSON-RPC Endpoint
# -------------------------------

@app.route("/mcp", methods=["POST"])
def mcp_endpoint():

    try:
        data = request.get_json(force=True)
        method = data.get("method")
        params = data.get("params", {})

        # Map method names to tool functions.
        tools = {
            "GetRelevantCode": get_relevant_code,
            "GetCodeFileContents": get_code_file_contents,
            "GetListOfCodeFiles": get_list_of_code_files
        }

        # Validate the method exists in the list of tools.
        if method not in tools:
            return jsonify({
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            })

        # Get the tool function and invoke it with parameters.
        params = data.get("params", {})
        tool_func = tools[method]
        result = invoke_tool(tool_func, params)

        # Return the result in JSON-RPC format.
        return jsonify({
            "jsonrpc": "2.0",
            "id": data.get("id"),
            "result": result
        })

    # Catch any unexpected errors.
    except Exception as e:
        return jsonify({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32603, "message": str(e)}
        })


# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    # Run the MCP server on localhost:5000.
    print("MCP server running at http://localhost:5000/mcp")
    app.run(host="0.0.0.0", port=5000)