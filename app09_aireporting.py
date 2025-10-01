import pandas as pd
import pandasql as psql
import re
import matplotlib.pyplot as plt
import streamlit as st

def load_data():
    sales = pd.read_csv("C:\myCODE\AgenticAIReporting\data\sales.csv")
    products = pd.read_csv("C:\myCODE\AgenticAIReporting\data\products.csv")
    stores = pd.read_csv("C:\myCODE\AgenticAIReporting\data\stores.csv")

    table_metadata = """Table: products; Columns: product_id	,product_name ,category	,brand	,price;
                        Table: sales; Columns: sale_id	,product_id	,store_id	,quantity	,sale_date;
                        Table: stores; Columns:store_id	,store_name	,city	,state	,region; """
    
    return sales, products, stores, table_metadata

import os
import langchain
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-nano",api_key=os.getenv("OPENAI_API_KEY"))

from langchain_google_genai import ChatGoogleGenerativeAI
llm_gai = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.getenv("GEMINI_API_KEY"), temperature=0.0)

#prompt template
from langchain.prompts import PromptTemplate
template = """You are a SQL data analyst. You help write sql code and visualize the results in charts.

Write a SQL query to answer the following question with only the tables/columns provided in metadata: 
Only return the SQL query starting with the Keyword "SQLQuery": Do not explain.
Suggest a visualization to display above Result, starting with the Keyword "Chart:".Do not explain. choose one of (bar, pie, line, scatter, table)
Add comment in query field selection to indicate 2 coordinates for the chart. use ("-- x coordinate", "-- y coordinate") Do not rename the query result field names.
Sample:
SQLQuery:
SELECT d.department_name as Department             -- x coordinate (Department)
        , count(e.employee_id) as Employees        -- y coordinate (Employee Count)
FROM employee e
JOIN department d on e.department_id=d.department_id

Chart: bar chart


Metadata = {metadata} 
Question: {question}"""
prompt = PromptTemplate(template=template, input_variables=["metadata","question"])

from langchain.schema import StrOutputParser
chain = prompt | llm | StrOutputParser()
chain_gai = prompt | llm_gai | StrOutputParser()

def invoke_llm_openai(question:str,table_metadata:str):
    response = chain.invoke({"question": question,"metadata":table_metadata})
    return response

def invoke_llm_gai(question:str,table_metadata:str):
    response = chain_gai.invoke({"question": question,"metadata":table_metadata})
    return response


def parse_sql_and_chart(parse_text):
    # Extract SQL query
    sql_query_match = re.search(r"sqlquery:\s*(select[\s\S]+?)\s*chart:", parse_text, re.IGNORECASE)
    sql_query = sql_query_match.group(1).strip() if sql_query_match else None

    # Extract chart suggestion
    chart_match = re.search(r"Chart:\s*(.*)", parse_text, re.IGNORECASE)
    chart_text = chart_match.group(1).strip() 
    print(chart_match)
    # Map common chart types
    if "bar" in chart_text:
        chart_type = "bar"
    elif "pie" in chart_text:
        chart_type = "pie"
    elif "line" in chart_text:
        chart_type = "line"
    elif "scatter" in chart_text:
        chart_type = "scatter"
    else:
        chart_type = "table"

    # Extract x and y coordinates from SQL comments
    # Look for "-- X coordinate (...)" or "-- Y coordinate (...)" in SELECT lines
    x_coord, y_coord = None, None
    select_lines = parse_text.splitlines()
    for line in select_lines:
        if "-- x coordinate" in line:
            if " as " in line:
                x_coord = line.split("as")[1].split()[0].strip().strip(',')
            elif " AS " in line:
                x_coord = line.split("AS")[1].split()[0].strip().strip(',')    
            else:
                x_coord = line.split(".")[1].split()[0].strip().strip(',')  
        if "-- y coordinate" in line:
            if " as " in line:
                y_coord = line.split("as")[1].split()[0].strip().strip(',') 
            elif " AS " in line:
                y_coord = line.split("AS")[1].split()[0].strip().strip(',')     
            else:
                y_coord = line.split(".")[1].split()[0].strip().strip(',')  

    return {
        "sql_query": sql_query,
        "chart": chart_type,
        "x_coordinate": x_coord,
        "y_coordinate": y_coord
    }

def query_validate(query: str):
    forbidden = ["insert","update","delete","drop","alter","create","truncate","attach"]
    for kw in forbidden:
        if re.search(r"\b"+kw+r"\b" , query.lower()):
            return False, "forbidden keyword found: "+kw
    return True, "Passed" 

def run_sql(query:str):
    try:
        result = psql.sqldf(query, globals())  # You can use locals() if inside a function
        print(f"[SQL] Query executed successfully. Rows returned: {len(result)}")
        return result
    except Exception as e:
        print(f"[ERROR] Failed to execute query:\n{query}\nReason: {e}")
        return None

def kpi_queries():
    total_sales_q = "SELECT SUM(quantity) as total_sales FROM sales;"
    total_revenue_q = """
        SELECT SUM(s.quantity * p.price)/1000000 as total_revenue
        FROM sales s
        JOIN products p ON s.product_id = p.product_id;
    """
    category_q = "SELECT COUNT(DISTINCT category) as products FROM products;"
    region_q = "SELECT COUNT(DISTINCT region) as stores FROM stores;"

    sales = run_sql(total_sales_q)["total_sales"].squeeze()
    revenue = run_sql(total_revenue_q)["total_revenue"].squeeze()
    category = run_sql(category_q)["products"].squeeze()
    region = run_sql(region_q)["stores"].squeeze()
    return sales, revenue, category,region

def load_kpi():
    sales, revenue, category,region = kpi_queries()
    col1, col2, col3, col4 = st.columns(4)    
    with col1:
        st.metric(label="🛒 Total Sales Volume", value=f"{sales:,}")
    with col2:
        st.metric(label="💰 Total Revenue in $M", value=f"${revenue:,.2f}")
    with col3:
        st.metric(label="📦 Product Category", value=f"{category:,}")
    with col4:
        st.metric(label="🏬 Region", value=f"{region:,}")


def display_bar(df, x: str, y: str):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df[x], df[y], color='skyblue')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis='x', rotation=45)
        return fig
    except Exception as e:
        st.error(f"[ERROR] display_bar failed: {e}")
        return None

def display_line(df, x: str, y: str):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df[x], df[y], marker='o', linestyle='-', color='blue')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        return fig
    except Exception as e:
        st.error(f"[ERROR] display_line failed: {e}")
        return None

def display_pie(df, x: str, y: str):
    try:
        if len(df) > 20:
            st.warning("Too many categories for pie chart. Showing top 20.")
            df = df.nlargest(20, y)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(df[y], labels=df[x], autopct='%1.1f%%',
               startangle=140, colors=plt.cm.Paired.colors)
        ax.axis("equal")
        return fig
    except Exception as e:
        st.error(f"[ERROR] display_pie failed: {e}")
        return None

def display_scatter(df, x: str, y: str):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[x], df[y], color='purple', alpha=0.7)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return fig
    except Exception as e:
        st.error(f"[ERROR] display_scatter failed: {e}")
        return None

def display_stacked_bar(df, x: str, y_cols: list):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom_vals = None
        for col in y_cols:
            ax.bar(df[x], df[col], bottom=bottom_vals, label=col, alpha=0.8)
            bottom_vals = df[col] if bottom_vals is None else bottom_vals + df[col]

        ax.set_xlabel(x)
        ax.set_ylabel("Values")
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        return fig
    except Exception as e:
        st.error(f"[ERROR] display_stacked_bar failed: {e}")
        return None

def display_table(df):
    try:
        df_show = df.head(20)
        fig, ax = plt.subplots(figsize=(8, min(0.5 * len(df_show), 6)))  # adjust height
        ax.axis("off")  # no axes

        # Create matplotlib table
        table = ax.table(
            cellText=df_show.values,
            colLabels=df_show.columns,
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)  # adjust cell scaling
        return fig
    except Exception as e:
        st.error(f"[ERROR] display_table failed: {e}")
        return None
    

def display_chart(chart: str, df, x: str, y, extra=None):
    if isinstance(df, pd.Series):
        df = df.reset_index()  # index becomes a column
        if df.shape[1] == 2:
            df.columns = [x, y]  # rename to match function call
        else:
            df.columns = [f"col{i}" for i in range(df.shape[1])]
    if df is None or df.empty:
        st.warning("No data to display.")
        return None
    chart = chart.lower()

    if chart == "bar":
        return display_bar(df, x, y)
    elif chart == "line":
        return display_line(df, x, y)
    elif chart == "pie":
        return display_pie(df, x, y)
    elif chart == "scatter":
        return display_scatter(df, x, y)
    else:
#        st.warning(f"Unknown chart type: {chart}")
        return display_table(df)  


################################################################################################
if __name__ == "__main__":
    @st.cache_data
    def load_all():
        sales, products, stores, table_metadata = load_data()
        return sales, products, stores, table_metadata

    sales, products, stores, table_metadata = load_all()

    # 🔹 App title
    st.title("AI Reporting - Sales")


    load_kpi()

#    st.write("""Available metadata:
#Table: products; Columns: product_id, product_name, category, brand, price;
#Table: sales; Columns: sale_id, product_id, store_id, quantity, sale_date;
#Table: stores; Columns: store_id, store_name, city, state, region;""")

    # 🔹 User input
    user_question = st.text_input("Ask questions about data:", 
                                "show volume of sales, revenue by product category")

    if st.button("Run Query"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                # LLM to SQL + chart
                llm_result = invoke_llm_gai(user_question, table_metadata)
  #              st.write("### AI Suggestion")
  #              st.code(llm_result, language="sql")

                parsed = parse_sql_and_chart(llm_result)

                valid, msg = query_validate(parsed["sql_query"])
  #              st.write("### Query Validation")
  #              st.write(f"Valid: {valid}, Message: {msg}")

                if valid:
                    try:
                        result = run_sql(parsed["sql_query"])
  #                      st.write("### Query Result (Top 20 rows)")
  #                      st.dataframe(result.head(20))

  #                      st.write("### Visualization")

                        fig = display_chart(parsed["chart"], result, parsed["x_coordinate"], parsed["y_coordinate"])
                        if fig:
                            st.pyplot(fig)

                        with st.expander("View SQL Query"):
                       #     st.code(parsed["sql_query"], language="sql")
                            st.code(llm_result, language="sql")

                    except Exception as e:
                        st.error(f"Query execution failed: {e}")
                else:
                    st.warning("Please modify the query or rephrase your question.")