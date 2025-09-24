import os
from langchain_openai import ChatOpenAI    
llm = ChatOpenAI(model="gpt-5-nano", api_key=os.getenv("OPENAI_API_KEY"))

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


tools = [wikipedia, math_tool]

from langchain.prompts import PromptTemplate
template = """Answer the question using the available tools.

You must use this format:

Question: the input question
Thought: your reasoning about what to do
Action Input: the input for the action
Observation: the result of the action
... (you can repeat Thought/Action/Action Input/Observation multiple times)
Final Answer: the final answer to the question

Begin!

Question: {input}
{agent_scratchpad}"""
prompt = PromptTemplate(template=template, input_variables=["input","agent_scratchpad"])  #{agent_scratchpad} required fills this with its internal thoughts + tool outputs during reasoning. also always need to ujse "input" no other naming

from langchain.agents import create_openai_tools_agent, AgentExecutor,create_react_agent, create_openai_functions_agent
# Initialize agent
agent = create_openai_functions_agent(llm,tools,prompt)
agent_executer = AgentExecutor(agent=agent, tools=tools,verbose=True,handle_parsing_errors=True)


# Run test query 
result = agent_executer.invoke({"input": "how far is london and new delhi"})  
print(result["output"])
