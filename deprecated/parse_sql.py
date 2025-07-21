# Databricks notebook source
# MAGIC %md
# MAGIC ## Simple Query Parser
# MAGIC
# MAGIC This parser will deconstuct a complex query into its sub-queries so each of those sub-queries can be separated out to be analyzed or converted.
# MAGIC
# MAGIC It will look for SELECT, cte's defined by WITH, IN or EXISTS to discover sub-queries.
# MAGIC
# MAGIC It will also can remove the columns if need be if a query has a long list of columns and replace them with a '*'.  This can be done to reduce query size for an LLM and analyze the columns separately.
# MAGIC
# MAGIC The script will put all of this information into a collection which can be iterated through to call an LLM to analyze or migrate the code and then it can be pieced back together at the end.
# MAGIC
# MAGIC So the following query:
# MAGIC
# MAGIC ```
# MAGIC with cte1 as (
# MAGIC         select cust_id, sum(sales) as sales_sum 
# MAGIC         from orders
# MAGIC         group by cust_id
# MAGIC     ),
# MAGIC     cte2 as (
# MAGIC         select cust_id, sum(expenses) as expenses
# MAGIC         from expenses
# MAGIC         group by cust_id
# MAGIC     )
# MAGIC     select cust_id, sales_sum, expenses
# MAGIC     from cte1
# MAGIC     inner join cte2 on cte1.cust_id = cte2.cust_id
# MAGIC     where cte1.cust_id in (select cust_id from customer_region where region = 'USA')
# MAGIC     and exists (select 1 from sales_person_region where region = 'NYC')
# MAGIC ```
# MAGIC
# MAGIC will be deconstructed to look like
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC Query 1 (cte1):
# MAGIC     select * 
# MAGIC         from orders
# MAGIC         group by cust_id
# MAGIC Columns:
# MAGIC   - cust_id
# MAGIC   - sum(sales) as sales_sum
# MAGIC ----------------------------------------
# MAGIC Query 2 (cte2):
# MAGIC     select *
# MAGIC         from expenses
# MAGIC         group by cust_id
# MAGIC Columns:
# MAGIC   - cust_id
# MAGIC   - sum(expenses) as expenses
# MAGIC ----------------------------------------
# MAGIC Query 3 (subquery):
# MAGIC     select * from customer_region where region = 'USA'
# MAGIC Columns:
# MAGIC   - cust_id
# MAGIC ----------------------------------------
# MAGIC Query 4 (subquery):
# MAGIC     select * from sales_person_region where region = 'NYC'
# MAGIC Columns:
# MAGIC   - 1
# MAGIC ----------------------------------------
# MAGIC Query 5 (main):
# MAGIC     select *
# MAGIC     from cte1
# MAGIC     inner join cte2 on cte1.cust_id = cte2.cust_id
# MAGIC     where cte1.cust_id in (<<query 2>>)
# MAGIC     and exists (<<query 3>>)
# MAGIC Columns:
# MAGIC   - cust_id
# MAGIC   - sales_sum
# MAGIC   - expenses
# MAGIC ----------------------------------------
# MAGIC main query:
# MAGIC with
# MAGIC     cte1 as (
# MAGIC         <<query 1>>
# MAGIC     ),
# MAGIC     cte2 as (
# MAGIC         <<query 2>>
# MAGIC     )
# MAGIC    <<query 5>>
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Install Libraries
# MAGIC %pip install sqlparse
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,SQL Parser
import re
from typing import List, Dict, Tuple

# def extract_columns(query: str) -> List[str]:
#     query = strip_comments(query)
#     query = query.strip()
#     columns = []

#     # Use regex to find all top-level SELECT ... FROM
#     pattern = re.compile(r'select\s+(.*?)\s+from', re.IGNORECASE | re.DOTALL)
#     matches = pattern.findall(query)

#     for cols in matches:
#         cols = cols.strip()
#         current = ''
#         depth = 0
#         for c in cols:
#             if c == ',' and depth == 0:
#                 columns.append(current.strip())
#                 current = ''
#             else:
#                 current += c
#                 if c == '(':
#                     depth += 1
#                 elif c == ')':
#                     depth -= 1
#         if current.strip():
#             columns.append(current.strip())

#     # Look for any columns that still have SELECT in them — pull them out as subqueries
#     expanded_columns = []
#     for col in columns:
#         if re.search(r'\bselect\b', col, re.IGNORECASE):
#             # run extract_subqueries on the column
#             rewritten, inner_subs = extract_subqueries(col)
#             expanded_columns.append(rewritten)
#             expanded_columns.extend(inner_subs)
#         else:
#             expanded_columns.append(col)

#     return expanded_columns

import re

def extract_columns(query: str) -> List[str]:
    query = strip_comments(query)
    select_match = re.search(r"select (.*?) from", query, re.IGNORECASE | re.DOTALL)
    if not select_match:
        return []

    col_string = select_match.group(1)
    columns = []
    current_col = ""
    paren_depth = 0

    for char in col_string:
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == ',' and paren_depth == 0:
            columns.append(current_col.strip())
            current_col = ""
            continue
        current_col += char

    if current_col.strip():
        columns.append(current_col.strip())

    return columns


def strip_comments(sql: str) -> str:
    # Remove multiline comments like /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

    # Remove inline and full-line comments starting with --
    sql = re.sub(r'--[^\n\r]*', '', sql)

    return sql

def replace_columns_with_star(query: str) -> str:
    query = strip_comments(query)
    query = query.strip()
    lower = query.lower()
    start = lower.find("select")
    if start == -1:
        return query

    depth = 0
    i = start + 6
    while i < len(query):
        if query[i] == '(':
            depth += 1
        elif query[i] == ')':
            depth -= 1
        elif lower[i:i+5] == ' from' and depth == 0:
            return query[:start + 6] + ' * ' + query[i:]
        i += 1
    return query

# def extract_subqueries(query: str) -> Tuple[str, List[str]]:
#     subqueries = []

#     def replace_subqueries(q: str) -> str:
#         output = ""
#         i = 0
#         n = len(q)

#         while i < n:
#             if q[i] == '(':
#                 start = i
#                 depth = 1
#                 i += 1
#                 content_start = i
#                 while i < n and depth > 0:
#                     if q[i] == '(':
#                         depth += 1
#                     elif q[i] == ')':
#                         depth -= 1
#                     i += 1
#                 content_end = i - 1
#                 content = q[content_start:content_end].strip()

#                 if re.search(r'\bselect\b', content, re.IGNORECASE):
#                     # Recursively rewrite inside first
#                     rewritten, inner_subs = extract_subqueries(content)
#                     subqueries.extend(inner_subs)

#                     # Only add the current expression if it starts with SELECT
#                     if re.match(r'^\s*select\b', content, re.IGNORECASE):
#                         subqueries.append(content)
#                         placeholder = f"<<subquery_{len(subqueries)}>>"
#                         output += f"({placeholder})"
#                     else:
#                         output += f"({rewritten})"
#                 else:
#                     output += f"({content})"
#             else:
#                 output += q[i]
#                 i += 1
#         return output

#     rewritten_query = query
#     prev_query = ""
#     while rewritten_query != prev_query:
#         prev_query = rewritten_query
#         rewritten_query = replace_subqueries(rewritten_query)

#     return rewritten_query, subqueries

def extract_subqueries(query: str) -> Tuple[str, List[str]]:
    subqueries = []

    def replace_subqueries(q: str) -> str:
        output = ""
        i = 0
        n = len(q)

        while i < n:
            if q[i] == '(':
                start = i
                depth = 1
                i += 1
                content_start = i
                while i < n and depth > 0:
                    if q[i] == '(':
                        depth += 1
                    elif q[i] == ')':
                        depth -= 1
                    i += 1
                content_end = i - 1
                content = q[content_start:content_end].strip()

                # Recursively handle inner content
                rewritten, inner_subs = extract_subqueries(content)
                subqueries.extend(inner_subs)

                if re.match(r'^\s*select\b', content, re.IGNORECASE):
                    subqueries.append(content)
                    placeholder = f"<<subquery_{len(subqueries)}>>"
                    output += f"({placeholder})"
                else:
                    output += f"({rewritten})"
            else:
                output += q[i]
                i += 1
        return output

    # Main first recursion on parentheses
    rewritten_query = query
    prev_query = ""
    while rewritten_query != prev_query:
        prev_query = rewritten_query
        rewritten_query = replace_subqueries(rewritten_query)

    # Now also check for top-level UNION etc
    union_pattern = re.compile(r'(select .*?)(?=(union|intersect|except|\Z))', re.IGNORECASE | re.DOTALL)
    union_matches = union_pattern.findall(rewritten_query)
    if len(union_matches) > 1:
        new_output = ""
        last_end = 0
        for i, (select_block, _) in enumerate(union_matches):
            subqueries.append(select_block.strip())
            placeholder = f"<<subquery_{len(subqueries)}>>"
            start_idx = rewritten_query.find(select_block, last_end)
            end_idx = start_idx + len(select_block)
            # Always insert a trailing space to prevent collisions like <<subquery>>UNION
            new_output += rewritten_query[last_end:start_idx] + placeholder + " "
            last_end = end_idx
        new_output += rewritten_query[last_end:]
        rewritten_query = new_output

    return rewritten_query, subqueries

def split_sql_view_full(sql: str, extract_columns_flag: bool = False) -> Tuple[List[Dict[str, object]], str, str]:
    sql = strip_comments(sql.strip())
    sql = sql.strip()
    sql = re.sub(r'\s+', ' ', sql, flags=re.IGNORECASE)
    mode = 'select'

    if sql.lower().startswith("with "):
        mode = 'with'
        sql_body = sql[5:].lstrip()
    elif sql.lower().startswith("select "):
        sql_body = sql
    else:
        raise ValueError("SQL must start with WITH or SELECT")

    queries = []
    idx = 0
    n = len(sql_body)
    cte_names = []

    if mode == 'with':
        while idx < n:
            match = re.match(r'(\w+)\s+as\s+\(', sql_body[idx:], re.IGNORECASE)
            if not match:
                break
            cte_name = match.group(1)
            cte_names.append(cte_name)
            start_idx = idx + match.end() - 1
            depth = 1
            end_idx = start_idx + 1
            while end_idx < n and depth > 0:
                if sql_body[end_idx] == '(':
                    depth += 1
                elif sql_body[end_idx] == ')':
                    depth -= 1
                end_idx += 1
            cte_block = sql_body[idx:end_idx].strip().rstrip(',')
            inner_query = re.match(rf'{cte_name}\s+as\s+\((.*)\)$', cte_block, re.IGNORECASE | re.DOTALL)
            inner_query_text = inner_query.group(1).strip() if inner_query else cte_block

            columns = extract_columns(inner_query_text) if extract_columns_flag else []
            starred = replace_columns_with_star(inner_query_text) if extract_columns_flag else inner_query_text

            queries.append({
                'name': cte_name,
                'type': 'cte',
                'columns': columns,
                'query': starred,
            })

            idx = end_idx
            while idx < n and sql_body[idx] in " ,\n\t":
                idx += 1
        main_query = sql_body[idx:].strip()
    else:
        main_query = sql_body

    rewritten_main, subqueries = extract_subqueries(main_query)

    for i, subquery in enumerate(subqueries):
        cols = extract_columns(subquery) if extract_columns_flag else []
        starred = replace_columns_with_star(subquery) if extract_columns_flag else subquery
        queries.append({
            'name': f"subquery_{i + 1}",
            'type': 'subquery',
            'columns': cols,
            'query': starred,
        })

    cols = extract_columns(main_query) if extract_columns_flag else []
    rewritten_main_starred = replace_columns_with_star(rewritten_main) if extract_columns_flag else rewritten_main

    if mode == 'with':
        full_main_query = "WITH " + ", ".join(
            [f"{name} AS (<<{name}>>)" for name in cte_names]
        ) + f" {rewritten_main_starred}"
    else:
        full_main_query = rewritten_main_starred

    queries.append({
        'name': 'main',
        'type': 'main',
        'columns': cols,
        'query': full_main_query,
    })

    return queries, mode, full_main_query

def print_full_queries(sql: str, extract_columns_flag: bool = False):
    parsed, mode, main_query = split_sql_view_full(sql, extract_columns_flag=extract_columns_flag)
    for i, entry in enumerate(parsed, 1):
        print(f"Query {i} ({entry['name']}):")
        print(f"    {entry['query']}")
        print("Columns:")
        for col in entry['columns']:
            print(f"  - {col}")
        print("-" * 40)

    if mode == 'with':
        print("main query:")
        ctes = [
            f"    {q['name']} as (\n        <<query {i + 1}>>\n    )"
            for i, q in enumerate(parsed[:-2]) if q['name'] != 'subquery'
        ]
        print("with\n" + ",\n".join(ctes))
        print(f"   <<query {len(parsed)}>>")
    else:
        print("main query:\n   <<query {len(parsed)}>>")


# COMMAND ----------

# DBTITLE 1,LLM Conversion Functions
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from pyspark.sql import SparkSession

def convert_query_to_databricks_sql(query: str, endpoint_name: str = "databricks-claude-sonnet-4"):
    w = WorkspaceClient()  # Initialize without timeout parameter, set timeout if supported later
    response = w.serving_endpoints.query(
        # name="databricks-claude-3-7-sonnet",
        name=endpoint_name,
        # name="llama-70b-code-converstion",
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM, content="You are a helpful assistant."
            ),
            ChatMessage(
                role=ChatMessageRole.USER, content=f"Please covert the following Oracle SQL query to Databricks SQL. Just return the query, no other content, including ```sql. I need a complete conversion, do not skip any lines:\n{query}"
            ),
        ]
    )
    return response

def get_split_sql_as_dataframe(query_string, extract_columns_flag: bool = False, endpoint_name: str = "databricks-claude-sonnet-4"):
    subqueries = []
    parsed, mode, main_query = split_sql_view_full(query_string, extract_columns_flag=extract_columns_flag)
    for i, entry in enumerate(parsed):
        subqueries.append(
            dict(
                name=entry['name'], 
                original=entry['query'],
                columns=entry.get('columns', [])
            )
        )

    # Convert the list of dictionaries to a Spark DataFrame
    subquery_df = spark.createDataFrame(subqueries)
    return subquery_df

from pyspark.sql import Row
import traceback

from pyspark.sql.types import StructType, StructField, StringType, ArrayType

def chunk_columns(columns, chunk_size=25):
    return [columns[i:i + chunk_size] for i in range(0, len(columns), chunk_size)]

# def convert_and_get_dataframe(
#     query_string, extract_columns_flag: bool = False, endpoint_name: str = "databricks-claude-sonnet-4",
#     test_mode: bool = False):

#     schema = StructType([
#         StructField("name", StringType(), True),
#         StructField("original", StringType(), True),
#         StructField("converted", StringType(), True),
#         StructField("columns", ArrayType(StringType()), True),
#         StructField("converted_columns", ArrayType(StringType()), True),
#         StructField("response_error", StringType(), True),
#         StructField("status", StringType(), True),
#     ])

#     converted = []
#     parsed, mode, main_query = split_sql_view_full(query_string, extract_columns_flag=extract_columns_flag)
    
#     for i, entry in enumerate(parsed):
#         try:
#             if not test_mode:
#                 response = convert_query_to_databricks_sql(entry['query'], endpoint_name)
#                 converted_query = response.choices[0].message.content
#             else:
#                 converted_query = entry['query']
#             status = "success"
#             response_error = ""
#         except Exception as e:
#             converted_query = ""
#             status = "failed"
#             response_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc(limit=2)}"

#         converted_columns_list = []
#         if extract_columns_flag and entry.get('columns'):
#             for chunk in chunk_columns(entry['columns'], 25):
#                 try:
#                     col_sql = "SELECT " + ", ".join(chunk) + " FROM dummy_table"
#                     if not test_mode:
#                         col_response = convert_query_to_databricks_sql(col_sql, endpoint_name)
#                         converted_sql = col_response.choices[0].message.content
#                     else:
#                         converted_sql = col_sql
#                     converted_cols = extract_columns(converted_sql)
#                     converted_columns_list.extend(converted_cols)
#                 except Exception as e:
#                     converted_columns_list.append("-- failed to convert columns: " + str(e))

#         converted.append(dict(
#             name=entry['name'],
#             original=entry['query'],
#             converted=converted_query,
#             columns=entry.get('columns', []),
#             converted_columns=converted_columns_list,
#             response_error=response_error,
#             status=status
#         ))

from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql import Row
import traceback

def convert_and_get_dataframe(
    query_string,
    extract_columns_flag: bool = False,
    columns_chunk_size: int = 25,
    endpoint_name: str = "databricks-claude-sonnet-4",
    test_mode: bool = False,
    target_table: str = None,
    failed_attempts: int = 5
):
    schema = StructType([
        StructField("name", StringType(), True),
        StructField("original", StringType(), True),
        StructField("converted", StringType(), True),
        StructField("columns", ArrayType(StringType()), True),
        StructField("converted_columns", ArrayType(StringType()), True),
        StructField("response_error", StringType(), True),
        StructField("status", StringType(), True),
    ])

    converted = []
    failed_count = 0
    parsed, mode, main_query = split_sql_view_full(query_string, extract_columns_flag=extract_columns_flag)

    for i, entry in enumerate(parsed):
        try:
            if not test_mode:
                response = convert_query_to_databricks_sql(entry['query'], endpoint_name)
                converted_query = response.choices[0].message.content
            else:
                converted_query = entry['query']
            status = "success"
            response_error = ""
        except Exception as e:
            failed_count += 1
            converted_query = ""
            status = "failed"
            response_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc(limit=2)}"

        converted_columns_list = []
        if extract_columns_flag and entry.get('columns'):
            for chunk in chunk_columns(entry['columns'], chunk_size=columns_chunk_size):
                try:
                    col_sql = "SELECT " + ", ".join(chunk) + " FROM dummy_table"
                    if not test_mode:
                        col_response = convert_query_to_databricks_sql(col_sql, endpoint_name)
                        converted_sql = col_response.choices[0].message.content
                    else:
                        converted_sql = col_sql
                    converted_cols = extract_columns(converted_sql)
                    converted_columns_list.extend(converted_cols)
                    status = "success"
                    response_error = ""
                except Exception as e:
                    converted_columns_list.append("-- failed to convert columns: " + str(e))
                    failed_count += 1
                    if failed_count >= failed_attempts:
                        print(f"Stopping early: {failed_count} failed attempts reached.")
                        status = "failed"
                        response_error = f"Stopping early: {failed_count} failed attempts reached. {type(e).__name__}: {str(e)}\n{traceback.format_exc(limit=2)}"

        converted.append(dict(
            name=entry['name'],
            original=entry['query'],
            converted=converted_query,
            columns=entry.get('columns', []),
            converted_columns=converted_columns_list,
            response_error=response_error,
            status=status
        ))

        converted_df = spark.createDataFrame(converted, schema=schema)

        # If catalog/schema provided, write to unity catalog
        try:
            if target_table:
                # print(f"Writing results to table: {target_table}")
                converted_df.write.mode("overwrite").format("delta").saveAsTable(target_table)
        except Exception as e:
            print(f"Error writing to table: {target_table}")

        if failed_count >= failed_attempts:
            print(f"Stopping early: {failed_count} failed attempts reached.")
            break

    return converted_df


# COMMAND ----------

# DBTITLE 1,Final Query Reassembly
from pyspark.sql.functions import col, when
import sqlparse

# def assemble_final_query(converted_df):
#     # Collect all rows
#     rows = converted_df.collect()

#     # Build a dict to store final rendered versions of each query
#     query_map = {}

#     # First pass: handle * replacement in each individual query (including subqueries)
#     for row in rows:
#         name = row['name']
#         query_text = row['converted'] if row['converted'] else row['original']

#         # Replace * with converted columns if available
#         if row['columns'] and row['converted_columns']:
#             all_converted_cols = " ".join(row['converted_columns'])
#             query_text = re.sub(r'(?i)(select\s+)\*', lambda m: m.group(1) + all_converted_cols, query_text, count=1)

#         query_map[name] = query_text

#     # Second pass: now plug subqueries into main query
#     main_query_text = query_map.get('main', '')

#     for name, subquery_text in query_map.items():
#         if name != 'main':
#             main_query_text = main_query_text.replace(f"<<{name}>>", subquery_text)

#     # Update DataFrame with final assembled main
#     updated_df = converted_df.withColumn(
#         "converted",
#         when(col("name") == "main", main_query_text).otherwise(col("converted"))
#     )

#     return updated_df.select("name", "original", "converted", "columns", "converted_columns")

from pyspark.sql.functions import col, when
import re

def assemble_final_query(converted_df, target_table: str = None):
    """
    - Replaces SELECT * with converted columns if present.
    - Recursively replaces <<subquery_x>> placeholders across all rows.
    - Updates the main query with fully resolved query.
    - If target_table provided, updates that Delta table's 'main' row.
    """

    rows = converted_df.collect()
    query_map = {}

    # Build initial map with SELECT * expansion
    for row in rows:
        name = row['name']
        query_text = row['converted'] if row['converted'] else row['original']

        if row['columns'] and row['converted_columns']:
            all_converted_cols = ", ".join(row['converted_columns'])
            query_text = re.sub(
                r'(?i)(select\s+)\*',
                lambda m: m.group(1) + all_converted_cols,
                query_text,
                count=1
            )

        query_map[name] = query_text

    # Recursive replacement of <<subquery_x>> placeholders everywhere
    changed = True
    while changed:
        changed = False
        for name, text in query_map.items():
            original_text = text
            for sub_name, sub_text in query_map.items():
                if sub_name != name:
                    text = text.replace(f"<<{sub_name}>>", sub_text)
            if text != original_text:
                changed = True
            query_map[name] = text

    # Update the local DataFrame with final main query
    final_main_query = query_map.get('main', '')
    updated_df = converted_df.withColumn(
        "converted",
        when(col("name") == "main", final_main_query).otherwise(col("converted"))
    )

    # If target_table specified, try to update that table's 'main' row
    if target_table:
        try:
            print(f"Updating existing table {target_table} with final assembled main query...")
            existing_df = spark.table(target_table)
            # Replace 'converted' where name == 'main'
            new_main_df = existing_df.withColumn(
                "converted",
                when(col("name") == "main", final_main_query).otherwise(col("converted"))
            )
            # Overwrite table
            new_main_df.write.mode("overwrite").format("delta").saveAsTable(target_table)
            print("Update complete.")
        except Exception as e:
            print(f"Error updating table {target_table}")

    return updated_df.select("name", "original", "converted", "columns", "converted_columns")

def get_main(converted_df):
    final_query_df = assemble_final_query(
        converted_df
    ).filter("name = 'main'")
    value = final_query_df.select("converted").collect()[0][0]
    return value

def prettify_final(query_string: str):
    # final_query_df = assemble_final_query(
    #     converted_df
    # ).filter("name = 'main'")
    # value = final_query_df.select("converted").collect()[0][0]

    # Format with sqlparse (keeps <<>> for any missing)
    prettified_value = sqlparse.format(query_string, reindent=True, keyword_case='upper')
    return prettified_value
