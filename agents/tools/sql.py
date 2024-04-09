import sqlite3
from pydantic.v1 import BaseModel
from typing import List
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)


def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"


# Adds a bit of info about the arguments, otherwise
# ChatGPT sees only __argN (N being 1, 2, 3 ... N)
class RunQueryArgsSchema(BaseModel):
    query: str


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema,
)


class DescribeTablesSchema(BaseModel):
    table_names: List[str]


def describe_tables(table_names: List[str]):
    c = conn.cursor()
    # "'users', 'orders', 'products'"
    tables = ", ".join("'" + table + "'" for table in table_names)
    c.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});"
    )
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)


describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables",
    func=describe_tables,
    args_schema=DescribeTablesSchema,
)
