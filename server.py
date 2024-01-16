from fastapi import FastAPI
from langserve import add_routes
from langchain.chains.base import Chain    
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import uvicorn

def start_server():
    # app = FastAPI(
    #     title="Test LangServe Server",
    #     version="1.0",
    #     description="A simple api server using LangServe"
    # )
    # add_routes(
    #     app,
    #     chain,
    #     path="/chain_route",
    # )
    # uvicorn.run(app, host="localhost", port=8000)
    # print("server started")
    # return

    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
    )

    add_routes(
        app,
        ChatOpenAI(),
        path="/openai",
    )

    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
    add_routes(
        app,
        prompt | model,
        path="/joke",
    )
    uvicorn.run(app, host="localhost", port=8000)
    return
