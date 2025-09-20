from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import load_tools

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain
from langchain_core.tools import Tool

# Load tokenizer + model
model_id = "google/vaultgemma-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=None,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Build Hugging Face pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=None,
    max_new_tokens=50,
    do_sample=False,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Load tools


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
llm_math = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful for answering math questions"
)

tools = [wikipedia, math_tool]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,   #ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run test query
#print(agent.run("what is 5 + 5"))

# Run test query with empty history
result = agent.invoke({"input": "what is 5 + 5", "chat_history": []})  #history for CONVERSATIONAL_REACT_DESCRIPTION
print(result["output"])
