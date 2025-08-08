import re
import sqlparse


def prettify_final(query_string: str):
    # Format with sqlparse (keeps <<>> for any missing)
    prettified_value = sqlparse.format(query_string, reindent=True, keyword_case='upper')
    return prettified_value


def strip_comments(sql: str) -> str:
    # Remove multiline comments like /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

    # Remove inline and full-line comments starting with --
    sql = re.sub(r'--[^\n\r]*', '', sql)

    return sql


def extract_and_replace_subqueries(sql_query, section:str="default", section_id:int=0):
    def parse(sql, base_idx=1):
        subqueries = []
        result = ""
        i = 0
        n = len(sql)
        subquery_counter = base_idx

        while i < n:
            # Copy normal text
            if sql[i] != '(':
                result += sql[i]
                i += 1
                continue

            # We hit a '(', skip whitespace to see if SELECT
            j = i + 1
            while j < n and sql[j].isspace():
                j += 1

            if j + 5 <= n and sql[j:j+6].lower() == 'select':
                # We found a (SELECT...
                paren_count = 1
                k = j + 6
                buffer = '(' + sql[j:j+6]

                while k < n and paren_count > 0:
                    buffer += sql[k]
                    if sql[k] == '(':
                        paren_count += 1
                    elif sql[k] == ')':
                        paren_count -= 1
                    k += 1

                # Recursive replacement inside this subquery
                inner_sql, inner_subs, next_counter = parse(buffer[1:-1], subquery_counter)
                subqueries.extend(inner_subs)

                # Store the current subquery properly as (placeholder, content)
                placeholder = f"<<subquery_{section}[{section_id}]_{next_counter}>>"
                subqueries.append((placeholder, inner_sql))
                result += f"({inner_sql})"
                subquery_counter = next_counter + 1
                i = k
            else:
                # Just a normal '('
                result += sql[i]
                i += 1

        return result, subqueries, subquery_counter

    clean_sql = strip_comments(sql_query)
    pretty_sql = prettify_final(clean_sql)
    # print(f"pretty:\n{pretty_sql}\n")
    final_sql, all_subqueries, _ = parse(pretty_sql)
    return final_sql, all_subqueries


def extract_all_subqueries_to_list(sql_query:str):
    final_sql, all_subqueries = extract_and_replace_subqueries(sql_query)
    return [{"id": q[0], "full_sub_query": q[1]} for q in all_subqueries]