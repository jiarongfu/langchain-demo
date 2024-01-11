
from langchain_community.chat_models.openai import ChatOpenAI

from agents.chat import get_agent

if __name__ == '__main__':
    # openai.api_key = os.environ['OPENAI_API_KEY']

    model = ChatOpenAI(temperature=0)
    agent = get_agent(model)

    while (user_input := input("Enter a question:")) != "EXIT":
        response = agent.invoke({"input": user_input})
        print(response)
