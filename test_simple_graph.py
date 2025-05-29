import asyncio
import os
import sys
import structlog
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

async def main():
   # Load environment variables
   load_dotenv()
   
   # Import dependencies
   from react_agent.mcp_client import MCPGatewayClient
   from react_agent.simple_graph import create_simple_graph
   from langchain_anthropic import ChatAnthropic
   
   # Initialize components
   llm = ChatAnthropic(
      model="claude-3-5-sonnet-20241022",
      temperature=0,
      api_key=os.getenv("ANTHROPIC_API_KEY")
   )
   
   mcp_client = MCPGatewayClient("http://localhost:8808")

   graph = await create_simple_graph(llm, mcp_client)
   #  print("✅ Simple graph created successfully")
    
    # Test 1: Basic context gathering
   # print("\n🧪 Test 1: Basic context gathering")
   # test_input = "Please create a new Python file called hello_world.py that prints 'Hello, World!' and run it to make sure it works."
   
   # try:
   #    result = await graph.run(test_input)
   #    print(f"✅ Test 1 completed successfully")
   #    print(f"Final messages count: {len(result['messages'])}")
   #    if result['messages']:
   #       print(f"Last message: {result['messages'][-1].content[:200]}...")
   # except Exception as e:
   #    print(f"❌ Test 1 failed: {e}")
   #    import traceback
   #    traceback.print_exc()
    
    # Test 2: GitHub branch creation workflow
   print("\n🧪 Test 2: Simple git branch creation test")
   branch_test_input = """
   Please create a new git branch called 'test-branch-123' in the repository dat04567/mutli-agent. 
   That's all I need you to do - just create the branch in my GitHub repository.
   """
   
   try:
      result = await graph.run(branch_test_input)
      print(f"✅ Test 2 completed successfully")
      print(f"Final messages count: {len(result['messages'])}")
      if result['messages']:
         print(f"Last message: {result['messages'][-1].content[:200]}...")
   except Exception as e:
      print(f"❌ Test 2 failed: {e}")
      import traceback
      traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
