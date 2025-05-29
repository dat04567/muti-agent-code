"""
Simplified LangGraph workflow that properly handles tool calls from MCP.
This addresses the core issue where the complex graph was converting tool calls to text.
"""
import asyncio
import structlog
from typing import Dict, Any, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from react_agent.state import State
from react_agent.agents.orchestrator import Orchestrator
from react_agent.agents.planner import Planner
from react_agent.agents.coder import Coder
from react_agent.mcp_client import MCPGatewayClient

log = structlog.get_logger()

class SimpleGraph:
    def __init__(self, llm: BaseChatModel, mcp_client: MCPGatewayClient):
        self.llm = llm
        self.mcp_client = mcp_client
        self.tools = []
        self.tool_map = {}
        
        # Load MCP tools and create routing tools
        self._load_tools()
        
        # Create agents with tools
        self.orchestrator = Orchestrator(llm, self.tools)
        self.planner = Planner(llm, self.tools)
        self.coder = Coder(llm, self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _load_tools(self):
        """Load MCP tools and create routing tools."""
        from react_agent.tools import _load_tools, ROUTING_TOOLS
        from react_agent import mcp_client
        
        # Set the MCP client globally 
        mcp_client.get_client("http://localhost:8808")
        
        # Load MCP tools
        mcp_tools = _load_tools()
        self.tools.extend(mcp_tools)
        self.tools.extend(ROUTING_TOOLS)
        
        # Create tool map for execution
        for tool in self.tools:
            self.tool_map[tool.name] = tool
    
    def _build_graph(self):
        """Build the simplified state graph."""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("coder", self._coder_node)
        workflow.add_node("tool_executor", self._tool_executor_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add edges based on routing decisions
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "planner": "planner",
                "coder": "coder", 
                "tools": "tool_executor",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "coder": "coder",
                "tools": "tool_executor",
                "orchestrator": "orchestrator",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "coder",
            self._route_from_coder,
            {
                "orchestrator": "orchestrator",
                "tools": "tool_executor",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "tool_executor",
            self._route_from_tools,
            {
                "orchestrator": "orchestrator",
                "planner": "planner", 
                "coder": "coder"
            }
        )
        
        return workflow.compile()
    
    async def _orchestrator_node(self, state: State) -> Dict[str, Any]:
        """Execute orchestrator agent."""
        log.info("Executing orchestrator node")
        result = await self.orchestrator.run(state)
        
        # Extract the response message from the result
        if isinstance(result, dict) and "messages" in result:
            response = result["messages"][0]  # Get the first message
        else:
            response = result
        
        # Handle the response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            log.info(f"Orchestrator made tool calls: {response.tool_calls}")
            state.pending_tool_calls = response.tool_calls
            state.current_agent = "orchestrator"
        else:
            log.info(f"Orchestrator response: {response.content}")
        
        # Add response to messages
        state.messages.append(response)
        
        return {"messages": state.messages, "pending_tool_calls": state.pending_tool_calls, "current_agent": state.current_agent}
    
    async def _planner_node(self, state: State) -> Dict[str, Any]:
        """Execute planner agent."""
        log.info("Executing planner node")
        result = await self.planner.run(state)
        
        # Extract the response message from the result
        if isinstance(result, dict) and "messages" in result:
            response = result["messages"][0]  # Get the first message
        else:
            response = result
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            log.info(f"Planner made tool calls: {response.tool_calls}")
            state.pending_tool_calls = response.tool_calls
            state.current_agent = "planner"
        else:
            log.info(f"Planner response: {response.content}")
        
        state.messages.append(response)
        
        return {"messages": state.messages, "pending_tool_calls": state.pending_tool_calls, "current_agent": state.current_agent}
    
    async def _coder_node(self, state: State) -> Dict[str, Any]:
        """Execute coder agent."""
        log.info("Executing coder node")
        result = await self.coder.run(state)
        
        # Extract the response message from the result
        if isinstance(result, dict) and "messages" in result:
            response = result["messages"][0]  # Get the first message
        else:
            response = result
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            log.info(f"Coder made tool calls: {response.tool_calls}")
            state.pending_tool_calls = response.tool_calls
            state.current_agent = "coder"
        else:
            log.info(f"Coder response: {response.content}")
        
        state.messages.append(response)
        
        return {"messages": state.messages, "pending_tool_calls": state.pending_tool_calls, "current_agent": state.current_agent}
    
    async def _tool_executor_node(self, state: State) -> Dict[str, Any]:
        """Execute pending tool calls."""
        log.info("Executing tool executor node")
        
        if not state.pending_tool_calls:
            log.warning("No pending tool calls to execute")
            return {"messages": state.messages}
        
        tool_messages = []
        
        for tool_call in state.pending_tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call.get("id", "")
            
            log.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    result = await tool.ainvoke(tool_args)
                    log.info(f"Tool {tool_name} result: {result}")
                    
                    tool_messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                        name=tool_name
                    ))
                else:
                    error_msg = f"Tool {tool_name} not found"
                    log.error(error_msg)
                    tool_messages.append(ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_id,
                        name=tool_name
                    ))
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                log.error(error_msg)
                tool_messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_id,
                    name=tool_name
                ))
        
        # Add tool results to messages
        state.messages.extend(tool_messages)
        
        # Clear pending tool calls
        state.pending_tool_calls = []
        
        return {"messages": state.messages, "pending_tool_calls": state.pending_tool_calls}
    
    def _route_from_orchestrator(self, state: State) -> str:
        """Route from orchestrator based on last message or tool calls."""
        if state.pending_tool_calls:
            # Check for routing tool calls
            for tool_call in state.pending_tool_calls:
                if tool_call["name"] == "route_to_planner":
                    return "planner"
                elif tool_call["name"] == "route_to_coder":
                    return "coder"
            return "tools"
        
        # Check last message for routing keywords
        if state.messages:
            last_content = state.messages[-1].content.lower()
            if "route_to_planner" in last_content:
                return "planner"
            elif "route_to_coder" in last_content:
                return "coder"
        
        return "end"
    
    def _route_from_planner(self, state: State) -> str:
        """Route from planner."""
        if state.pending_tool_calls:
            for tool_call in state.pending_tool_calls:
                if tool_call["name"] == "route_to_coder":
                    return "coder"
                elif tool_call["name"] == "route_to_orchestrator":
                    return "orchestrator"
            return "tools"
        
        if state.messages:
            last_content = state.messages[-1].content.lower()
            if "route_to_coder" in last_content:
                return "coder"
        
        return "end"
    
    def _route_from_coder(self, state: State) -> str:
        """Route from coder."""
        if state.pending_tool_calls:
            for tool_call in state.pending_tool_calls:
                if tool_call["name"] == "route_to_orchestrator":
                    return "orchestrator"
            return "tools"
        
        if state.messages:
            last_content = state.messages[-1].content.lower()
            if "route_to_orchestrator" in last_content:
                return "orchestrator"
        
        return "end"
    
    def _route_from_tools(self, state: State) -> str:
        """Route back to the agent that made the tool calls."""
        return state.current_agent or "orchestrator"
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """Run the graph with user input."""
        log.info(f"Starting graph execution with input: {user_input}")
        
        # Initialize state
        state = State(
            messages=[HumanMessage(content=user_input)],
            pending_tool_calls=[],
            current_agent="orchestrator"
        )
        
        # Execute the graph
        result = await self.graph.ainvoke(state)
        
        log.info("Graph execution completed")
        return result


# Factory function to create the simple graph
async def create_simple_graph(llm: BaseChatModel, mcp_client: MCPGatewayClient) -> SimpleGraph:
    """Create and return a simple graph instance."""
    return SimpleGraph(llm, mcp_client)


# LangGraph Studio support
def create_graph_for_studio():
    """Create graph for LangGraph Studio visualization."""
    import os
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import AIMessage
    from react_agent.state import State
    
    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Create a minimal graph for studio visualization without MCP tools
    workflow = StateGraph(State)
    
    # Add simple nodes for visualization
    def orchestrator_demo(state: State):
        return {"messages": state.messages + [AIMessage(content="Orchestrator processing...")]}
    
    def planner_demo(state: State):
        return {"messages": state.messages + [AIMessage(content="Planner creating plan...")]}
    
    def coder_demo(state: State):
        return {"messages": state.messages + [AIMessage(content="Coder implementing solution...")]}
    
    def tool_executor_demo(state: State):
        return {"messages": state.messages + [AIMessage(content="Executing tools...")]}
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_demo)
    workflow.add_node("planner", planner_demo)
    workflow.add_node("coder", coder_demo)
    workflow.add_node("tool_executor", tool_executor_demo)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add simple routing for visualization
    workflow.add_conditional_edges(
        "orchestrator",
        lambda state: "planner",
        {"planner": "planner"}
    )
    
    workflow.add_conditional_edges(
        "planner", 
        lambda state: "coder",
        {"coder": "coder"}
    )
    
    workflow.add_conditional_edges(
        "coder",
        lambda state: "tool_executor", 
        {"tool_executor": "tool_executor"}
    )
    
    workflow.add_conditional_edges(
        "tool_executor",
        lambda state: "end",
        {"end": END}
    )
    
    return workflow.compile()
