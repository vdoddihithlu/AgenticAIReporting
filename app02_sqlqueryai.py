
#  model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-nano",api_key="")

#prompt template
from langchain.prompts import PromptTemplate
template = """You are a SQL data analyst. you help write sql code
You have Tables/columns listed in metadata:
{metadata}

Write a SQL query to answer the following question with only the tables/columns provided in metadata:
Question: {question}

Only return the SQL query. Do not explain."""
prompt = PromptTemplate(template=template, input_variables=["metadata","question"])

# Run
table_metadata = """Table: Products; Columns: product_id	,product_name ,category	,brand	,price
                    Table: Sales; Columns: sale_id	,product_id	,store_id	,quantity	,sale_date
                    Table: Stores; Columns:store_id	,store_name	,city	,state	,region """

from langchain.schema import StrOutputParser
chain = prompt | llm | StrOutputParser()

def getAnswer(question:str):
    final_prompt = prompt.format(question=question,metadata=table_metadata)
    response = chain.invoke({"question": question,"metadata":table_metadata})
    return response

################################################################################################
if __name__ == "__main__":
    question = "show volume of sales and revenue by product category"

    response = getAnswer(question)
    print({"Question":question})
    print({"Answer": response})


#{'Question': 'show volume of sales and revenue by product category'}
#{'Answer': 'SELECT\n  p.category,\n  SUM(s.quantity) AS total_quantity,\n  SUM(s.quantity * COALESCE(p.price, 0)) AS total_revenue\nFROM Sales s\nJOIN Products p ON s.product_id = p.product_id\nGROUP BY p.category\nORDER BY total_revenue DESC;'}

#{'Question': 'show volume of sales and revenue by product category'}
#{'Answer': 'SELECT p.Category,\n       COUNT(*) AS Volume,\n       SUM(s.SalePrice) AS Revenue'
#'\nFROM Product p'
#'\nJOIN Sales s ON p.product_id = s.product_id'
#'\nGROUP BY p.Category'
#'\nORDER BY Revenue DESC;'}