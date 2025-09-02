from flask import Flask, request, jsonify
import inspect

app = Flask(__name__)

# -------------------------------
# Tool Implementations
# -------------------------------


def get_test_failure() -> str:

    # Simulated test failure output. In a production environment, this method would dynamically get the failure for a specified test.
    return """
        1) [chromium] › tests/vending-machine.spec.ts:33:5 › insert money ────────────────────────────────

        Error: Timed out 5000ms waiting for expect(locator).toContainText(expected)

        Locator: getByTestId('money-display')
        Expected string: "$0.25"
        Received string: "$-0.25"
        Call log:
            - Expect "toContainText" with timeout 5000ms
            - waiting for getByTestId('money-display')
            9 × locator resolved to <div class="money-display" data-testid="money-display">$-0.25</div>
                - unexpected value "$-0.25"
    """


def get_test_code() -> str:
    # Returns the code of the failing test. In a production environment, this method would use RAG
    # to return just the relevant test code and not necessarily an entire test file.
    return open("../ecommerce-website/tests/vending-machine.spec.ts").read().strip()


def get_app_code() -> str:
    # Returns the application code. In a production environment, this method would use RAG
    # to return just the relevant app code and not necessarily an entire app file.
    return open("../ecommerce-website/src/App.tsx").read().strip()


# -------------------------------
# JSON-RPC Endpoint
# -------------------------------


import inspect

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


@app.route("/mcp", methods=["POST"])
def mcp_endpoint():

    try:
        data = request.get_json(force=True)
        method = data.get("method")
        params = data.get("params", {})

        # Map method names to tool functions.
        tools = {
            "GetTestFailure": get_test_failure,
            "GetTestCode": get_test_code,
            "GetAppCode": get_app_code
        }

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

        return jsonify({
            "jsonrpc": "2.0",
            "id": data.get("id"),
            "result": result
        })

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