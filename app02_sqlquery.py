from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

# Load tokenizer + model
model_id = "google/vaultgemma-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=False)  
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Build Hugging Face pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=50,          # small limit
    do_sample=False,        # deterministic
    temperature=0.1,            # no randomness
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

#prompt template
template = """You are a data analyst. you help write sql code
Sample question: how many employees in each department?
Answer: select d.Department_name, count(e.Employee_ID) from Employee e join Department d on d.department_id=e.department_id group by d.Department_name

You have Tables/columns listed in metadata:
{metadata}

Write a SQL query to answer the following question with only the tables/columns provided in metadata:
Question: {question}

Only return the SQL query. Do not explain."""
prompt = PromptTemplate(template=template, input_variables=["metadata","question"])

# Run
#table_metadata = """Table:employees, Columns:emp_id,Name,department_id, Salary
#                    Table:departments, Columns:department_id,department_name"""
table_metadata = """Table: Product; Columns: product_id, Name, Category, Price
                    Table: Sales; Columns:product_id, Sales_date, Customer_name  """

def getAnswer(question:str):
    final_prompt = prompt.format(question=question,metadata=table_metadata)
    response = llm.invoke(final_prompt)
    print({"Full Question":final_prompt})
    return response.replace(final_prompt,"")

################################################################################################
if __name__ == "__main__":
    question = "show distinct count of products"

    response = getAnswer(question)
    print({"Question":question})
    print({"Answer": response.replace("\nAnswer:","")})
