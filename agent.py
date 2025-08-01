from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################

import os
import pandas as pd
import yaml

import os
import pandas as pd

def build_example_string(csv_path=None):
    """
    Loads a CSV and builds a formatted string of examples.
    - If csv_path is provided and exists, uses that.
    - If csv_path is None or doesn't exist, falls back to EXAMPLES_FILE env variable.
    - Ensures the final chosen path exists before loading.
    """
    # Check if provided path is valid
    if csv_path and os.path.exists(csv_path):
        final_path = csv_path
    else:
        # Try environment variable
        env_file = os.getenv("ORACLE_TO_DATABRICKS_EXAMPLE_FILE")
        if env_file and os.path.exists(env_file):
            final_path = env_file
        else:
            print("CSV file not found. Neither provided path nor environment variable 'EXAMPLES_FILE' is valid.")
            return ""

    print(f"!!!!! final_path: {final_path}")
    df = pd.read_csv(final_path)

    output = []
    for _, row in df.iterrows():
        formatted = (
            f"Oracle:\n{row['oracle_query']}\n"
            f"INCORRECT (Do NOT do this):\n{row['incorrect_query']}\n"
            f"CORRECT (DO this):\n{row['correct_query']}\n\n"
        )
        output.append(formatted)

    return "".join(output).strip()

with open("agent_config.yaml", "r") as file:
    prompts = yaml.safe_load(file)

# LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
LLM_ENDPOINT_NAME = prompts["llm_endpoint_name"]
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# system_prompt = """Please covert the following Oracle SQL query to Databricks SQL. Just return the query, no other content, including ```sql. If you see any sql that is wrapped in << >>, for example <<subquery_1>>, assume it is valid sql and leave it as is.  I need a complete conversion, do not skip any lines"""
system_prompt = prompts["oracle_to_databricks_system_prompt"]
example_file = prompts.get("example_file", None)
example_string = build_example_string(example_file)
print(f"!!!!!! example_string: {example_string}")
system_prompt = system_prompt.format(examples=example_string)
print(f"\n\nsystem_prompt\n\n{system_prompt}")

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
uc_tool_names = []
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)


# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# # for details
#
# # TODO: Add vector search indexes as tools or delete this block
# vector_search_tools = [
#         VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# ]
# tools.extend(vector_search_tools)


#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# def set_prompt_query_pretext(query_pretext: str = ""):
#     system_prompt = system_prompt.replace("{query_pretext}", query_pretext)


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # print(messages)
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
