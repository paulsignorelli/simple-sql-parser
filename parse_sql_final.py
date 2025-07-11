
import re

def extract_subqueries(query: str):
    subqueries = []

    def replace_subqueries(q: str) -> str:
        output = ""
        i = 0
        n = len(q)
        while i < n:
            c = q[i]
            if c == '(':
                depth = 1
                start = i
                i += 1
                while i < n and depth > 0:
                    if q[i] == '(':
                        depth += 1
                    elif q[i] == ')':
                        depth -= 1
                    i += 1
                content = q[start+1:i-1]
                rewritten = replace_subqueries(content)
                select_idx = content.lower().find("select")
                if select_idx != -1:
                    select_part = content[select_idx:]
                    prefix = content[:select_idx]
                    subqueries.append(select_part)
                    placeholder = f"<<subquery_{len(subqueries)}>>"
                    output += f"{prefix}{placeholder}"
                else:
                    output += f"({rewritten})"
            else:
                output += c
                i += 1
        return output

    new_query = replace_subqueries(query)
    return new_query, subqueries


def smart_column_split(select_clause):
    cols = []
    current = ''
    depth = 0
    for char in select_clause:
        if char == ',' and depth == 0:
            cols.append(current.strip())
            current = ''
        else:
            current += char
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
    if current:
        cols.append(current.strip())
    return cols


def extract_columns_from_single_select(select_query):
    select_idx = select_query.upper().find("SELECT")
    from_idx = select_query.upper().find("FROM")
    if select_idx == -1 or from_idx == -1:
        return []
    select_clause = select_query[select_idx + len("SELECT"):from_idx]
    return smart_column_split(select_clause)


def parse_query_columns(query: str):
    all_columns = []
    parts = re.split(r'\bUNION\s+ALL\b', query, flags=re.IGNORECASE)
    for part in parts:
        columns = extract_columns_from_single_select(part)
        all_columns.extend(columns)
    return all_columns
