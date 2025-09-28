
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
import torch


# Load tokenizer + model
model_id = "google/vaultgemma-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id,
    device_map=None,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Build Hugging Face pipeline   
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=None,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Load tools

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

from langchain.chains import LLMMathChain
from langchain_core.tools import Tool
llm_math = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful for answering math questions"
)

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
webpage = WebBaseLoader("https://python.langchain.com/docs/integrations/tools/")
document=webpage.load()
documentchunks = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(document)
vectordb = FAISS.from_documents(documentchunks,HuggingFaceEmbeddings())
retriver = vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
docsearchtool = create_retriever_tool(retriver,"doc","search for information about langchain tool")


tools = [wikipedia, math_tool, docsearchtool]

from langchain.prompts import PromptTemplate
template = """Answer the question using the available tools. {tools}

You must use this format:

Question: the input question
Thought: your reasoning about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input for the action
Observation: the result of the action
... (you can repeat Thought/Action/Action Input/Observation multiple times)
Final Answer: the final answer to the question

Begin!

Question: {input}
{agent_scratchpad}"""
prompt = PromptTemplate(template=template, input_variables=["input","agent_scratchpad","tools", "tool_names"])  #{agent_scratchpad} required fills this with its internal thoughts + tool outputs during reasoning. also always need to ujse "input" no other naming

from langchain.agents import create_openai_tools_agent, AgentExecutor,create_react_agent
# Initialize agent
# agent = create_openai_tools_agent(llm,tools,prompt)
agent = create_react_agent(llm,tools,prompt)
agent_executer = AgentExecutor(agent=agent, tools=tools,verbose=True,handle_parsing_errors=True)


# Run test query 
result = agent_executer.invoke({"input": "what 2 + 2"})  
print(result["output"])
