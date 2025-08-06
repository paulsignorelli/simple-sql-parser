# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/build-genai-apps.html) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC
# MAGIC import os
# MAGIC import pandas as pd
# MAGIC import yaml
# MAGIC
# MAGIC import os
# MAGIC import pandas as pd
# MAGIC
# MAGIC def build_example_string(csv_path=None):
# MAGIC     """
# MAGIC     Loads a CSV and builds a formatted string of examples.
# MAGIC     - If csv_path is provided and exists, uses that.
# MAGIC     - If csv_path is None or doesn't exist, falls back to EXAMPLES_FILE env variable.
# MAGIC     - Ensures the final chosen path exists before loading.
# MAGIC     """
# MAGIC     # Check if provided path is valid
# MAGIC     if csv_path and os.path.exists(csv_path):
# MAGIC         final_path = csv_path
# MAGIC     else:
# MAGIC         # Try environment variable
# MAGIC         env_file = os.getenv("ORACLE_TO_DATABRICKS_EXAMPLE_FILE")
# MAGIC         if env_file and os.path.exists(env_file):
# MAGIC             final_path = env_file
# MAGIC         else:
# MAGIC             print("CSV file not found. Neither provided path nor environment variable 'EXAMPLES_FILE' is valid.")
# MAGIC             return ""
# MAGIC
# MAGIC     print(f"!!!!! final_path: {final_path}")
# MAGIC     df = pd.read_csv(final_path)
# MAGIC
# MAGIC     output = []
# MAGIC     for _, row in df.iterrows():
# MAGIC         formatted = (
# MAGIC             f"Oracle:\n{row['oracle_query']}\n"
# MAGIC             f"INCORRECT (Do NOT do this):\n{row['incorrect_query']}\n"
# MAGIC             f"CORRECT (DO this):\n{row['correct_query']}\n\n"
# MAGIC         )
# MAGIC         output.append(formatted)
# MAGIC
# MAGIC     return "".join(output).strip()
# MAGIC
# MAGIC with open("agent_config.yaml", "r") as file:
# MAGIC     prompts = yaml.safe_load(file)
# MAGIC
# MAGIC # LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC LLM_ENDPOINT_NAME = prompts["llm_endpoint_name"]
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC # system_prompt = """Please covert the following Oracle SQL query to Databricks SQL. Just return the query, no other content, including ```sql. If you see any sql that is wrapped in << >>, for example <<subquery_1>>, assume it is valid sql and leave it as is.  I need a complete conversion, do not skip any lines"""
# MAGIC # system_prompt = prompts["oracle_to_databricks_system_prompt"]
# MAGIC system_prompt_file = prompts["oracle_to_databricks_system_prompt_file"]
# MAGIC with open(system_prompt_file, "r") as f:
# MAGIC     system_prompt = f.read()
# MAGIC
# MAGIC # example_file = prompts.get("example_file", None)
# MAGIC # example_string = build_example_string(example_file)
# MAGIC # print(f"!!!!!! example_string: {example_string}")
# MAGIC # system_prompt = system_prompt.format(examples=example_string)
# MAGIC
# MAGIC print(f"\n\nsystem_prompt\n\n{system_prompt}")
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC uc_tool_names = []
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC
# MAGIC # # (Optional) Use Databricks vector search indexes as tools
# MAGIC # # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# MAGIC # # for details
# MAGIC #
# MAGIC # # TODO: Add vector search indexes as tools or delete this block
# MAGIC # vector_search_tools = [
# MAGIC #         VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # ]
# MAGIC # tools.extend(vector_search_tools)
# MAGIC
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC # def set_prompt_query_pretext(query_pretext: str = ""):
# MAGIC #     system_prompt = system_prompt.replace("{query_pretext}", query_pretext)
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         # print(messages)
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

# COMMAND ----------

import re
from typing import List, Dict

def extract_sql_sections(message: str) -> List[Dict[str, str]]:
    """
    Extracts all SQL code blocks (```sql ... ```) and associates them 
    with the nearest preceding section heading (e.g., 'Oracle', 'Databricks').
    Also appends a dictionary containing the full message.
    
    Returns:
        [
            {"section": "Original Oracle SQL", "sql": "NVL(...)"},
            {"section": "Converted Databricks SQL", "sql": "COALESCE(...)"},
            {"section": "FULL_MESSAGE", "sql": "<entire message>"}
        ]
    """
    results = []
    
    # Find all headings and their positions
    headings = [(m.start(), m.group(1).strip()) 
                for m in re.finditer(r"##\s*(.*?)\s*:", message)]
    
    # Find all SQL code blocks
    for sql_match in re.finditer(r"```sql\s*([\s\S]*?)\s*```", message, re.IGNORECASE):
        sql_start = sql_match.start()
        sql_code = sql_match.group(1).strip()
        
        # Find the closest heading before this SQL block
        section_name = "Unknown"
        for pos, heading in reversed(headings):
            if pos < sql_start:
                section_name = heading
                break
        
        results.append({
            "section": section_name,
            "sql": sql_code
        })
    
    # Add full message as a separate entry
    results.append({"section": "FULL_MESSAGE", "sql": message.strip()})
    
    return results


# # Example usage
# msg_content = """
# # Oracle to Databricks SQL Conversion

# ## Original Oracle SQL:
# ```sql
# NVL(
#     (
#         SELECT emp_id FROM Employees WHERE ROWNUM=1
#     )
# , 2)


# COMMAND ----------

import re

query = f"""


WITH
DeptStats AS (
    SELECT
        DepartmentID,
        SUM(Salary) AS TotalDeptSalary,
        COUNT(*) AS NumEmployees,
        AVG(Salary) AS AvgDeptSalary
    FROM
        Employees
    GROUP BY
        DepartmentID
),
EmpProjects AS (
    SELECT
        EmployeeID,
        COUNT(ProjectID) AS ProjectsCompleted,
        MAX(CompletedDate) AS LastProjectDate,
        YEAR(MAX(CompletedDate)) AS LastProjectYear
    FROM
        Projects
    WHERE
        Status = 'Completed'
    GROUP BY
        EmployeeID
)
SELECT
    e.EmployeeID,
    UPPER(e.Name) AS Name,
    d.Name AS Department,
    CASE
        WHEN e.Salary > (
            SELECT AVG(Salary)
            FROM Employees
            WHERE DepartmentID = e.DepartmentID
        ) THEN CONCAT('Above Average (', CAST(e.Salary AS VARCHAR), ')')
        ELSE 'Average or Below'
    END AS SalaryStatus,
    -- some remark here,
    CASE
        WHEN    rsm.investment_type = 'BL'
            AND NVL (psah.acrd_cd, 'N') NOT IN ('Y', 'V') -- story 897300
        THEN
            NVL (
                (SELECT wacoupon
                   FROM stg_wso_pos_acr_ame
                  WHERE     portfolio_fund_id = psah.cal_dt
                        AND asofdate = psah.cal_dt
                        AND asset_primaryud = psah.asset_id
                        AND rec_typ_cd = 'POS'
                        AND ROWNUM=1),
                0)
        ELSE
            psah.int_rt
    END
        AS pos_int_it,  
    ep.ProjectsCompleted,
    YEAR(e.HireDate) AS HireYear,
    MONTH(e.HireDate) AS HireMonth,
    COALESCE(ep.LastProjectYear, 'N/A') AS LastProjectYear
FROM
    Employees e
    JOIN Departments d ON e.DepartmentID = d.DepartmentID
    LEFT JOIN EmpProjects ep ON e.EmployeeID = ep.EmployeeID
WHERE
    e.EmployeeID IN (
        SELECT
            e2.EmployeeID
        FROM
            Employees e2
            JOIN Departments d2 ON e2.DepartmentID = d2.DepartmentID
            LEFT JOIN EmpProjects ep2 ON e2.EmployeeID = ep2.EmployeeID
        WHERE
            e2.Salary > (
                SELECT AVG(Salary)
                FROM Employees
                WHERE DepartmentID = e2.DepartmentID
            )
    )
UNION ALL
SELECT
    NULL AS EmployeeID,
    NULL AS Name,
    d.Name AS Department,
    CONCAT('Department Total: ', CAST(ds.TotalDeptSalary AS VARCHAR)) AS SalaryStatus,
    ds.NumEmployees AS ProjectsCompleted,
    NULL AS HireYear,
    NULL AS HireMonth,
    NULL AS LastProjectYear,
    SELCT region_id FROM regoins WHERE ROWNUM=1,

'col1' as col1,
'col2' as col2,
'col3' as col3,
'col4' as col4,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),'x')
ELSE 0 END as col5,
myfunc_from(0.2260155996892257) as col6,
'col7' as col7,
'col8' as col8,
'col9' as col9,
'col10' as col10,
'col11' as col11,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees JOIN dept ON Employees.dept_id = dept.dept_id
WHERE DepartmentID = ds.DepartmentID),0)
ELSE 0 END as col12,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),2)
ELSE 0 END as col13,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),'x')
ELSE 0 END as col14,
'col15' as col15,
'col16' as col16,
'col17' as col17,
'col18' as col18,
'col19' as col19,
'col20' as col20,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),'x')
ELSE 0 END as col21,
'col22' as col22,
'col23' as col23,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees JOIN dept ON Employees.dept_id = dept.dept_id
WHERE DepartmentID = ds.DepartmentID),0)
ELSE 0 END as col24,
'col25' as col25,
'col26' as col26,
'col27' as col27,
'col28' as col28,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees JOIN dept ON Employees.dept_id = dept.dept_id
WHERE DepartmentID = ds.DepartmentID),0)
ELSE 0 END as col29,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),2)
ELSE 0 END as col30,
'col31' as col31,
'col32' as col32,
'col33' as col33,
'col34' as col34,
'col35' as col35,
myfunc_from(0.23160372299847443) as col36,
'col37' as col37,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),'x')
ELSE 0 END as col38,
'col39' as col39,
'col40' as col40,
'col41' as col41,
'col42' as col42,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),2)
ELSE 0 END as col43,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees JOIN dept ON Employees.dept_id = dept.dept_id
WHERE DepartmentID = ds.DepartmentID),0)
ELSE 0 END as col44,
'col45' as col45,
myfunc_fromchar(0.29850520193203534) as col46,
'col47' as col47,
'col48' as col48,
'col49' as col49,
'col50' as col50,
'col51' as col51,
CASE WHEN ds.TotalDeptSalary > 100000 AND ds.TotalDeptSalary < 200000 AND NVL(a.emp_class, 'N') NOT IN ('A', 'B') -- story 1234
THEN NVL((SELECT id FROM Employees WHERE DepartmentID = ds.DepartmentID),2)
ELSE 0 END as col52,
'col53' as col53,
'col54' as col54,
'col55' as col55,
'col56' as col56,
'col57' as col57,
'col58' as col58,
'col59' as col59
FROM
    DeptStats ds
    JOIN Departments d ON ds.DepartmentID = d.DepartmentID;

"""
response = AGENT.predict({"messages": [{"role": "user", "content": f"Convert the following sql to Databricks\n\n{query}"}]})

sql_block = None
for msg in response.messages:
    if msg.role == "assistant":
        # pattern = (
        #     r"##\s*Converted\s+Databricks\s+SQL:\s*"
        #     r"(?:```sql\s*([\s\S]*?)\s*```|sql\s*([\s\S]*?)(?:\n##|\Z))"
        # )
        # match = re.search(pattern, msg.content, re.DOTALL)
        # if match:
        #     sql_block = match.group(1)
        #     break
        sql_block = extract_sql_sections(msg.content)
        break

print(sql_block)

# COMMAND ----------

from pyspark.sql.functions import explode, col
df = spark.createDataFrame([(sql_block,)], ["response"])
df_exploded = df.select(explode(col("response")).alias("item"))
df_final = df_exploded.select(
    col("item.section").alias("section"),
    col("item.sql").alias("sql")
)
display(df_final)

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "What is 5+5 in python"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) or leverages [external functions](https://docs.databricks.com/generative-ai/agent-framework/external-connection-tools.html), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import LLM_ENDPOINT_NAME, tools
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        # TODO: If the UC function includes dependencies like external connection or vector search, please include them manually.
        # See the TODO in the markdown above for more information.
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "SELECT * FROM Employees GROUP BY DepartmentID"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/mlflow3/genai/eval-monitor)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC Evaluate your agent with one of our [predefined LLM scorers](https://docs.databricks.com/mlflow3/genai/eval-monitor/predefined-judge-scorers), or try adding [custom metrics](https://docs.databricks.com/mlflow3/genai/eval-monitor/custom-scorers).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness

eval_dataset = [
    {
        "inputs": {
            "messages": [
                {
                    "role": "system",
                    "content": "Please covert the following Oracle SQL query to Databricks SQL. Just return the query, no other content, including ```sql. If you see any sql that is wrapped in << >>, for example <<subquery_1>>, assume it is valid sql and leave it as is.  I need a complete conversion, do not skip any lines"
                },
                {
                    "role": "user",
                    "content": "SELECT * FROM Employees GROUP BY DepartmentID"
                }
            ]
        },
        "expected_response": None
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda messages: AGENT.predict({"messages": messages}),
    scorers=[RelevanceToQuery(), Safety()], # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://docs.databricks.com/machine-learning/model-serving/model-serving-debug.html#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = ""
schema = ""
model_name = ""
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://docs.databricks.com/generative-ai/deploy-agent.html) for details
