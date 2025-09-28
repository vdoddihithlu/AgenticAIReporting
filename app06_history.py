from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch


from app00_reusablefunction import pretty_print_history

# Load tokenizer + model
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50
)


# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

#prompt template
template = """ {question} """
prompt = PromptTemplate(template=template, input_variables=["question"])

################# chains
from langchain.schema import StrOutputParser
base_chain = prompt | llm | StrOutputParser()

# define a history store function
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# simple in-memory history --dictonary
_history_store = {}

# function to create new key for the seesionid if not there and store and return the chat history.
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _history_store:
        # using default memory implementation
        _history_store[session_id] = ChatMessageHistory()
    return _history_store[session_id]

# wrap the chain with history to send context
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)


def getAnswer(question:str, session_id: str = "default"):
       
    response = chain_with_history.invoke({"question": question}  #other parameter like metadata are passed along here. no need to pass in chain_with_history
                                        ,config={"configurable": {"session_id": session_id}})
    #print("Chat History so far:", get_history(session_id).messages)
    pretty_print_history(get_history(session_id).messages)
    return response

################################################################################################
if __name__ == "__main__":
    # Same session_id will keep history
    session_id = "default"

    q1 = "explain gravity"
    r1 = getAnswer(q1,session_id)
    print({"Question": q1, "Answer": r1})

    q2 = "who discoverd it?"
    r2 = getAnswer(q2,session_id)   # same session_id, so history is passed
    print({"Question": q2, "Answer": r2})

    q3 = "and when?"
    r3 = getAnswer(q3,session_id)   # continues using the same history
    print({"Question": q3, "Answer": r3})
