from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load tokenizer + model
model_id = "google/vaultgemma-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=False)  
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=None, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Build Hugging Face pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=None,        #"auto"
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
Answer: select d.Department_name, count(e.Employee_ID) as Employees
from Employee e 
join Department d on d.department_id=e.department_id 
group by d.Department_name

You have Tables/columns listed in metadata:
{metadata}

Write a SQL query to answer the following question with only the tables/columns provided in metadata:
Question: {question}

Only return the SQL query. Do not explain."""
prompt = PromptTemplate(template=template, input_variables=["metadata","question"])

################# chains
base_chain = prompt | llm

# define a history store function
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# simple in-memory history
_history_store = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _history_store:
        # using default memory implementation
        from langchain_community.chat_message_histories import ChatMessageHistory
        _history_store[session_id] = ChatMessageHistory()
    return _history_store[session_id]

# wrap with history
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)


def getAnswer(question:str):
    table_metadata = """Table: Product; Columns: product_id, Name, Category, Price
                    Table: Sales; Columns:product_id, Sales_date, Customer_name  """
    final_prompt = prompt.format(question=question,metadata=table_metadata) 
    
    response = chain_with_history.invoke({"question": question, "metadata": table_metadata}
                                        ,config={"configurable": {"session_id": "default"}})
    print({"Full Question":final_prompt})
    return response.replace(final_prompt,"")

################################################################################################
if __name__ == "__main__":
    question = "the answer is wrong"
    response = getAnswer(question)
    print({"Question":question})
    print({"Answer": response.replace("\nAnswer:","")})
