from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_core.language_models import BaseChatModel
from langchain.agents import AgentExecutor

from tools.meteo import get_current_temperature
from tools.wikipedia import search_wikipedia

def get_agent(model:BaseChatModel):
    tools = [get_current_temperature, search_wikipedia]
    functions = [format_tool_to_openai_function(f) for f in tools]
    model_with_func = model.bind(functions=functions)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        # MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    # memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
    agent_chain = RunnablePassthrough.assign(
        agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model_with_func | OpenAIFunctionsAgentOutputParser()
    # return AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)
    return AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

