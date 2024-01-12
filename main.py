import os
from uuid import uuid4
from langchain_community.chat_models.openai import ChatOpenAI
from langsmith import Client

from agents.chat import get_agent

if __name__ == '__main__':
    # openai.api_key = os.environ['OPENAI_API_KEY']
    unique_id = uuid4().hex[0:8]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    model = ChatOpenAI(temperature=0)
    agent = get_agent(model)

    while (user_input := input("Enter a question(type EXIT to stop): ")) != "EXIT":
        response = agent.invoke({"input": user_input})
        print(f"Response:\n{response}")
