# MCP Workshop For Beginners

This workshop is for people who are new to AI code and new to MCP.

You do **not** need to understand the face-recognition logic in this repository. That part already exists. Your job is to rebuild the **MCP layer**: the small pieces of code that expose functionality from a server and call that functionality from a client.

By the end of this workshop, you will have practiced the core MCP ideas:

- creating an MCP server
- exposing tools
- exposing a resource
- connecting a client to an MCP server
- listing tools from a client
- reading a resource from a client
- calling a tool from a client
- connecting an agent to MCP tools

## Which Folders To Use

Use these workshop folders:

- Server: `siamese_mcp_uncomplete/`
- Client: `siamese_mcp_client_umcomplete/`
- Agent: `agent_uncomplete/`

These folders already contain missing pieces marked with `TODO` comments and `NotImplementedError(...)`.

## What You Do Not Need To Touch

Please leave these files alone unless you are curious:

- `siamese_mcp_uncomplete/src/siamese_mcp/face_service.py`
- `siamese_mcp_client_umcomplete/src/siamese_mcp_client/models.py`
- Docker and database files

Those files contain the business logic and data setup. In this workshop, we are only learning the MCP basics.

## Big Picture

Think of the application in two halves:

1. The **server** publishes capabilities.
2. The **client** discovers and uses those capabilities.

In this workshop:

- the server exposes two tools and one resource
- the client connects to the server
- the client lists tools
- the client reads the resource
- the client calls the tools

## Before You Start

If you want to run the full app locally, start from the repository root:

```bash
docker compose up -d --build
```

That should start the server at:

```text
http://127.0.0.1:8000/mcp
```

## Recommended Order

Follow the tasks in this order:

1. Rebuild the server tools.
2. Rebuild the server resource.
3. Rebuild the client connection.
4. Rebuild tool discovery and resource reading.
5. Rebuild client-side tool calls.
6. Rebuild the simple CLI helpers.
7. Complete the agent wiring.

## Part 1: Server Tools

Open:

`siamese_mcp_uncomplete/src/siamese_mcp/server.py`

You need to implement:

- `register_face_image_tool()`
- `search_face_image_tool()`

### What these functions should do

These are thin wrapper functions around existing Python logic.

That means:

- they receive arguments from MCP
- they optionally log a message with `ctx.info(...)`
- they call an existing Python function
- they return the result

The existing business logic already exists here:

- `register_face_image(...)`
- `search_face_image(...)`

Your job is just to connect MCP to those functions.

### MCP Pattern To Copy

This is the basic shape of an MCP tool wrapper:

```python
@mcp.tool(name="example_tool", description="Example description")
async def example_tool(value: str, ctx: Context) -> dict[str, Any]:
    await ctx.info(f"Running tool for {value}")
    return {"value": value}
```

### What to look for in your own code

When rebuilding the two server tools, ask yourself:

- Am I accepting the same arguments that are already in the function signature?
- Am I forwarding all arguments to the existing service function?
- Am I returning a Python `dict`?
- Am I using `ctx.info(...)` for a simple progress message?

### Done when

You are done with this section when:

- the server exposes both tools
- a client can list both tool names
- the tool call returns JSON-like structured data

### Helpful sources

- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Microsoft Learn: Build an MCP server into a Python app](https://learn.microsoft.com/en-us/azure/app-service/tutorial-ai-model-context-protocol-server-python)
- [Microsoft MCP for Beginners](https://aka.ms/PythonMCP/repo)

## Part 2: Server Resource

Still in:

`siamese_mcp_uncomplete/src/siamese_mcp/server.py`

You need to implement:

- `get_service_info()`

### What a resource is

A resource is read-only information that the client can fetch.

In this workshop, the resource should return metadata about the service, for example:

- the MCP path
- the transport type
- accepted file extensions

### MCP Pattern To Copy

```python
@mcp.resource("service://example", mime_type="application/json")
def get_example_resource() -> dict[str, Any]:
    return {
        "transport": "streamable-http",
        "path": "/mcp",
    }
```

### What to look for in your own code

Ask yourself:

- Does the function return a normal Python dictionary?
- Does that dictionary describe the service?
- Is the function decorated with `@mcp.resource(...)`?

### Done when

You are done with this section when:

- the client can read `service://face-recognition`
- the output is JSON-like metadata

### Helpful sources

- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Microsoft Learn: Build an MCP server into a Python app](https://learn.microsoft.com/en-us/azure/app-service/tutorial-ai-model-context-protocol-server-python)

## Part 3: Client Connection

Open:

`siamese_mcp_client_umcomplete/src/siamese_mcp_client/client.py`

You need to implement:

- `connect()`

### What this function should do

This function opens the connection from the client to the MCP server.

At a high level it should:

1. create an HTTP client
2. open a streamable HTTP connection
3. create a `ClientSession`
4. initialize the session
5. store the session and connection state on `self`

### MCP Pattern To Copy

This is the general pattern:

```python
stack = AsyncExitStack()

http_client = create_mcp_http_client(headers=self.headers)
await stack.enter_async_context(http_client)

read_stream, write_stream, session_id_callback = await stack.enter_async_context(
    streamable_http_client(self.server_url, http_client=http_client)
)

session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
initialize_result = await session.initialize()
```

You will need to adapt that pattern to the class that already exists in the file.

### What to look for in your own code

Ask yourself:

- Am I cleaning up properly if something fails?
- Am I storing `_stack`, `_session`, `_session_id_callback`, and `_initialize_result`?
- Does `async with SiameseMcpClient(...)` now work?

### Done when

You are done with this section when:

- the client connects successfully
- the client can call `list_tools()` afterwards

### Helpful sources

- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Python SDK docs](https://modelcontextprotocol.github.io/python-sdk/)

## Part 4: Client Discovery

Still in:

`siamese_mcp_client_umcomplete/src/siamese_mcp_client/client.py`

You need to implement:

- `list_tools()`
- `get_service_info()`

### What these functions should do

`list_tools()` should ask the MCP server which tools it exposes.

`get_service_info()` should read the resource `service://face-recognition` and turn the returned JSON into the `ServiceInfo` model.

### Example pattern

```python
session = self._require_session()
result = await session.list_tools()
return result.tools
```

And for a resource:

```python
session = self._require_session()
result = await session.read_resource("service://example")
payload = self._extract_json_from_resource(result)
return SomeModel.model_validate(payload)
```

### What to look for in your own code

Ask yourself:

- Am I using `_require_session()` first?
- Am I returning `result.tools` for tool discovery?
- Am I using `_extract_json_from_resource(...)` before validating the model?

### Done when

You are done with this section when these commands work from the client folder:

```bash
python -m siamese_mcp_client.cli tools
python -m siamese_mcp_client.cli service
```

### Helpful sources

- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Microsoft Learn: Using MCP tools with agents](https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools)

## Part 5: Client Tool Calls

Still in:

`siamese_mcp_client_umcomplete/src/siamese_mcp_client/client.py`

You need to implement:

- `register_face_image()`
- `search_face_image()`

### What these functions should do

Each function should:

1. get the active session
2. build an `arguments` dictionary
3. call the correct MCP tool by name
4. check whether the tool call failed
5. extract JSON from the tool response
6. validate that JSON into the correct response model

### Example pattern

```python
session = self._require_session()

arguments = {
    "filename": filename,
    "image_jpeg_base64": image_jpeg_base64,
}

result = await session.call_tool("some_tool_name", arguments=arguments)
if result.isError:
    raise SomeError("Tool failed")

payload = self._extract_json_from_tool_result(result)
return SomeResponseModel.model_validate(payload)
```

You do not need to invent new response parsing code. Most helpers already exist in the file.

### What to look for in your own code

Ask yourself:

- Am I calling the correct tool name constant?
- Did I include all expected arguments?
- Do I raise `SiameseMcpToolError` if `result.isError` is true?
- Do I validate into `RegisterFaceResponse` or `SearchFaceResponse`?

### Done when

You are done with this section when:

- the client can call the server tools
- successful calls become Pydantic models
- failed calls raise a useful error

### Helpful sources

- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Python SDK docs](https://modelcontextprotocol.github.io/python-sdk/)

## Part 6: CLI Tasks

Open:

`siamese_mcp_client_umcomplete/src/siamese_mcp_client/cli.py`

You must implement:

- `_run_tools()`
- `_run_service()`

### What these functions should do

These are simple command-line helpers, but they are also a good exercise because they show how the client is used in a real Python program.

They should:

- create the client with `async with`
- call the correct client method
- print the result in a helpful way
- return `0`

### What is already done for you

Most of the CLI wiring already exists.

You do **not** need to build the full CLI from scratch.

The parser is already configured in `build_parser()`, and each command already points to a handler:

- `tools` -> `_run_tools()`
- `service` -> `_run_service()`

That means your job is only to fill in the body of the async handler functions.

### Best way to approach it

Before writing code, look at the two handlers that are already complete in the same file:

- `_run_register()`
- `_run_search()`

Those functions already show the pattern you should follow:

1. open the client with `async with`
2. call one client method
3. print the returned value
4. return `0`

For this exercise, the main difference is:

- `_run_tools()` returns a list of tools, so you will probably loop over them
- `_run_service()` returns one model object, so you can print it as JSON

### Step-by-step for `_run_tools()`

In `_run_tools(args)`:

1. create the client with `async with SiameseMcpClient(server_url=args.server_url) as client:`
2. call `await client.list_tools()`
3. store the result in a variable, for example `tools`
4. loop through the tools
5. print each tool name and description
6. return `0`

The shape will be similar to this:

```python
async with SomeClient(server_url=args.server_url) as client:
    tools = await client.list_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description or ''}")

return 0
```

### Step-by-step for `_run_service()`

In `_run_service(args)`:

1. create the client with `async with SiameseMcpClient(server_url=args.server_url) as client:`
2. call `await client.get_service_info()`
3. store the result in a variable, for example `service_info`
4. print it
5. return `0`

Because `service_info` is a Pydantic model, the easiest output is JSON:

```python
async with SomeClient(server_url=args.server_url) as client:
    service_info = await client.get_service_info()

print(service_info.model_dump_json(indent=2))
return 0
```

### Done when

These commands work:

```bash
python -m siamese_mcp_client.cli tools
python -m siamese_mcp_client.cli service
```

And the output should make sense:

- `tools` should print tool names and descriptions
- `service` should print JSON with service metadata

### Helpful sources

- [Python argparse tutorial](https://docs.python.org/3/howto/argparse.html)
- [Python asyncio high-level API](https://docs.python.org/3/library/asyncio.html)
- [Pydantic models and serialization](https://docs.pydantic.dev/latest/concepts/serialization/)
- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## Part 7: Agent Challenge

Open:

`agent_uncomplete/agent.py`

This part is for learners who want to see how MCP tools can be used from an agent.

You do **not** need to build a full agent system from scratch.

The main job is to fill in the missing `TODO` values.

### What you need to complete

Fill in these parts:

- the agent instructions
- the `name` and `url` inside `MCPStreamableHTTPTool(...)`
- the `project_endpoint`, `deployment_name`, and `credential` inside `AzureOpenAIResponsesClient(...)`
- the `name` and `instructions` inside `Agent(...)`

### What each part means

For the instructions:

- describe what kind of assistant the agent should be
- explain which MCP tools it can use
- explain how it should handle image paths
- explain how it should format responses

For `MCPStreamableHTTPTool(...)`:

- `name` should be a clear human-readable tool connection name
- `url` should point to the Siamese MCP endpoint

For `AzureOpenAIResponsesClient(...)`:

- `project_endpoint` should come from your Azure AI project setup
- `deployment_name` should match your Azure OpenAI deployment
- `credential` should be the Azure credential object already created in the `async with` block

For `Agent(...)`:

- `name` should clearly describe the agent
- `instructions` should use the instruction text you created earlier in the file

### Good order to solve it

1. Read the full `agent_uncomplete/agent.py` file first.
2. Find every `TODO`.
3. Write the agent instructions.
4. Fill in the MCP tool connection values.
5. Fill in the Azure client values.
6. Fill in the final agent values.
7. Run the script and try one simple prompt.

### Hints

- The MCP endpoint in this workshop is usually `http://127.0.0.1:8000/mcp`.
- The credential is not a string. It should be the Azure credential object from the async context.
- The agent instructions are important because they influence when the agent calls tools and how it explains results.
- This project uses the server-side `/shared/` folder for image tests, so the agent should know how to turn a filename into `/shared/<filename>`.

### Done when

You are done with this section when:

- there are no `TODO` placeholders left in `agent_uncomplete/agent.py`
- the script starts without configuration errors
- the agent can answer a prompt and use the MCP server

### Helpful sources

- `agent/agent.py`
- `README.md`
- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Microsoft Learn: Using MCP tools with agents](https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools)

## How To Test Yourself

From the repository root:

```bash
docker compose up --build
```

From the client workshop folder:

```bash
cd siamese_mcp_client_umcomplete
python -m siamese_mcp_client.cli tools
python -m siamese_mcp_client.cli service
```

### Prepare your test data

Before you test image matching, make sure you set up the data correctly:

- add at least one face image to the database
- use the person's name as the image filename
- place the image you want to compare in the `shared/` folder

The main idea is:

- the database should already contain a known face
- that known face should have been added with a filename that identifies the person
- the image you want to compare should be available in `shared/` so the server can access it

Example:

- if you add `elonmusk.jpg` to the database, the filename represents the known person
- if you want to test a new picture against that person, put that new picture in `shared/`

### Test the MCP server

If you also finished the tool-calling methods, try:

```bash
python -m siamese_mcp_client.cli register --image /absolute/path/to/person_name.jpg
python -m siamese_mcp_client.cli search --image /absolute/path/to/comparison_image.jpg
```

If you are testing through the agent:

- make sure the comparison image is inside `shared/`
- ask the agent to register or search using the filename
- the agent should translate that filename into `/shared/<filename>` before calling the MCP tool

## Final Goal

When you are finished, you should be able to explain these three things in your own words:

1. What is an MCP tool?
2. What is an MCP resource?
3. How does a client connect to a server and call a tool?

If you can explain those clearly and your code runs, then you completed the workshop successfully.
