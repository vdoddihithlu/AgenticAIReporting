import pprint
from langchain_core.messages import HumanMessage, AIMessage

def pretty_print_history(history, question=None, answer=None):
    print("######################################################################################\n" + "="*80)
    print("Chat History so far:")
    print("="*80)

    for i, msg in enumerate(history):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"\n[{i+1}] {role}Message:")
        pprint.pprint(vars(msg), indent=2, width=100)

    if question and answer:
        print("\n" + "-"*80)
        print("Latest Q&A:")
        print("-"*80)
        print(f"Question: {question}")
        print("Answer:")
        print(answer)
        print("-"*80)
