from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

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
template = """You are a answering machine. Answer Question: {question} ."""
prompt = PromptTemplate(template=template, input_variables=["question"])

def getAnswer(question:str):
    final_prompt = prompt.format(question=question)
    
    chain = prompt|llm 
    
    response = chain.invoke({"question": question})
    print({"Full Question":final_prompt})
    return response.replace(final_prompt,"")

################################################################################################
if __name__ == "__main__":
    question = "show unique count of products"
    response = getAnswer(question)
    print({"Question":question})
    print({"Answer": response.replace("\nAnswer:","")})
