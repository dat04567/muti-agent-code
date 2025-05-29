from typing import Dict, Any, List
import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableSequence
from react_agent.state import State

log = structlog.get_logger()

class BaseAgent:
    def __init__(self, name: str, prompt_template: str, llm: BaseChatModel, tools: List = None):
        self.name = name
        self.llm = llm
        if tools:
            self.llm = self.llm.bind_tools(tools)
            
        # Convert the template into system and human messages
        system_message = SystemMessage(content=prompt_template)
        human_template = HumanMessagePromptTemplate.from_template("{all_messages}")
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            human_template
        ])
        self.chain = RunnableSequence(self.prompt | self.llm)

    async def run(self, state: State) -> Dict[str, Any]:
        last_message = ""
        all_messages = ""
        
        if state.messages:
            last_message_obj = state.messages[-1]
            last_message = last_message_obj.content
            all_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in state.messages])
        else:
            # Provide a default message if there are no messages yet
            all_messages = "Let's begin the task."
        
        response = await self.chain.ainvoke({
            "last_message": last_message,
            "all_messages": all_messages
        })
        
        log.info(f"{self.name} full response object: {response}")
        log.info(f"{self.name} response type: {type(response)}")
        if hasattr(response, 'additional_kwargs'):
            log.info(f"{self.name} additional kwargs: {response.additional_kwargs}")
        if hasattr(response, 'tool_calls'):
            log.info(f"{self.name} tool calls: {response.tool_calls}")
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        log.info(f"{self.name} response content: {response_content}")

        # Return the full AIMessage response instead of converting to dict
        # This preserves tool calls and all message attributes
        return {
            "messages": [response],  # Keep the full AIMessage object
            "current_agent": self.name
        }
