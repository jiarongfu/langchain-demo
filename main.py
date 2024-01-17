import os
from uuid import uuid4
from langchain_community.chat_models.openai import ChatOpenAI
from langchain import callbacks
from langsmith import Client
from traceloop.sdk import Traceloop

from agents.chat import get_agent
from server import start_server
from test import run_test

if __name__ == '__main__':

    USE_SERVER = False
    RUN_TEST = False

    # openai.api_key = os.environ['OPENAI_API_KEY']
    # uuid = uuid4()
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # os.environ["LANGCHAIN_PROJECT"] = f"Test- {uuid.hex[0:8]}"
    # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    # client = Client()
  
    Traceloop.init(disable_batch=True, api_key="454306f7aaea52339dd2e6f515e8e6a746202f80d37a77bae1a984509c7d28cbdf886e17fbcc1f92368cec5fb8394736")
    model = ChatOpenAI(temperature=0)
    agent = get_agent(model)

    if RUN_TEST:
        def new_agent():
            return get_agent(model)
        run_test(new_agent)
    else:
        if USE_SERVER:
            start_server()
        else:
            while (user_input := input("Enter a question(type EXIT to stop): ")) != "EXIT":
                with callbacks.collect_runs() as cb:
                    response = agent.invoke({"input": user_input}, include_run_info=True)  
                    run_id = cb.traced_runs[0].id

                print(f"Response:\n{response}")
                # user_feedback = input("Please rate the answer on a scale of 1-5, with 5 being very satisfied: ")
                # client.create_feedback(
                #     run_id=run_id, 
                #     key="user_rating", 
                #     score=int(user_feedback), 
                #     comment="User rates the answer on a scale of 1-5, with 5 being very satisfied"
                # )
