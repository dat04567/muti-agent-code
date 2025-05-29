# Multi-Agent LangGraph with MCP Integration

A sophisticated multi-agent system built with LangGraph that integrates with Model Context Protocol (MCP) for advanced tool execution and workflow automation.

## 🚀 Features

- **Multi-Agent Architecture**: Orchestrator, Planner, Coder, and Tool Executor agents working in coordination
- **MCP Integration**: 38+ tools including filesystem operations, Git commands, and GitHub integration
- **LangGraph Studio Support**: Visual workflow representation and debugging
- **Async Tool Execution**: Non-blocking tool calls with proper error handling
- **Real-time Collaboration**: Agents can route tasks between each other dynamically

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orchestrator  │───▶│     Planner     │───▶│      Coder      │
│                 │    │                 │    │                 │
│ - Analyzes task │    │ - Creates plan  │    │ - Implements    │
│ - Routes agents │    │ - Breaks down   │    │ - Executes code │
│ - Coordinates   │    │   requirements  │    │ - Tests results │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  Tool Executor  │
                    │                 │
                    │ - File ops      │
                    │ - Git commands  │
                    │ - GitHub API    │
                    │ - System calls  │
                    └─────────────────┘
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd cuoiKyNLP3
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start MCP Gateway Server**
   ```bash
   cd gateway
   python -m mcp_gateway.server
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
GOOGLE_API_KEY=your_google_key_here

# MCP Gateway
MCP_GATEWAY_URL=http://localhost:8808

# Optional: GitHub Integration
GITHUB_TOKEN=your_github_token_here
```

### MCP Tools Configuration

The system automatically loads tools from:
- **Filesystem Tools**: read_file, write_file, list_directory, create_directory
- **Git Tools**: git_branch, git_commit, git_push, git_status
- **GitHub Tools**: create_repository, create_issue, create_pull_request
- **System Tools**: run_command, get_environment

## 🚦 Quick Start

### 1. Basic Usage

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from src.react_agent.simple_graph import create_simple_graph
from src.react_agent.mcp_client import MCPGatewayClient

async def main():
    # Initialize LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    # Initialize MCP client
    mcp_client = MCPGatewayClient("http://localhost:8808")
    
    # Create graph
    graph = await create_simple_graph(llm, mcp_client)
    
    # Run workflow
    result = await graph.run("Create a new Python script to analyze data")
    print(result)

asyncio.run(main())
```

### 2. LangGraph Studio

Start the visual interface:

```bash
langgraph dev
```

Then open `http://localhost:2024` in your browser to see the workflow visualization.

### 3. Testing

Run the test suite:

```bash
# Test simple graph
python test_simple_graph.py

# Test brand workflow
python gateway/brand_tools.py

# Run unit tests
python -m pytest tests/
```

## 📁 Project Structure

```
/
├── src/react_agent/           # Main agent system
│   ├── agents/               # Individual agent implementations
│   │   ├── orchestrator.py   # Task coordination agent
│   │   ├── planner.py        # Planning agent
│   │   ├── coder.py          # Implementation agent
│   │   └── base_agent.py     # Base agent class
│   ├── simple_graph.py       # Simplified workflow
│   ├── graph.py              # Complex workflow (legacy)
│   ├── mcp_client.py         # MCP gateway client
│   ├── tools.py              # Tool definitions
│   ├── state.py              # Workflow state management
│   └── prompts.py            # Agent prompts
├── gateway/                  # MCP Gateway Server
│   ├── src/mcp_gateway/      # Gateway implementation
│   └── config.json           # Gateway configuration
├── tests/                    # Test suite
├── static/                   # Documentation assets
└── langgraph.json           # LangGraph Studio config
```

## 🔄 Workflow Examples

### Example 1: Create a Brand Analysis Tool

```python
result = await graph.run("""
Create a brand analysis tool that:
1. Creates a new Git branch 'brand-analysis'
2. Writes a Python script for brand sentiment analysis
3. Commits the changes with proper documentation
""")
```

### Example 2: Code Review Automation

```python
result = await graph.run("""
Review the recent code changes and:
1. Check for coding standards compliance
2. Run automated tests
3. Generate a summary report
4. Create GitHub issues for any problems found
""")
```

## 🧪 Testing & Development

### Running Tests

```bash
# All tests
python -m pytest

# Integration tests only
python -m pytest tests/integration_tests/

# Unit tests only
python -m pytest tests/unit_tests/

# With coverage
python -m pytest --cov=src/react_agent
```

### Development Mode

```bash
# Start in development mode with auto-reload
langgraph dev --reload

# Enable blocking calls for debugging
langgraph dev --allow-blocking

# Custom port
langgraph dev --port 3000
```

## 🔍 Monitoring & Debugging

### Logs

The system uses structured logging with different levels:

```python
import structlog
log = structlog.get_logger()

# View real-time logs
tail -f logs/agent.log
```

### LangGraph Studio

- **Workflow Visualization**: See agent interactions in real-time
- **State Inspection**: Debug state transitions
- **Tool Call Monitoring**: Track tool executions
- **Performance Metrics**: Monitor execution times

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code style
- Add type hints to all functions
- Write comprehensive tests for new features
- Update documentation for any API changes

## 📚 API Reference

### SimpleGraph Class

```python
class SimpleGraph:
    async def run(self, user_input: str) -> Dict[str, Any]:
        """Execute workflow with user input"""
        
    def _load_tools(self) -> None:
        """Load MCP tools and routing tools"""
        
    def _build_graph(self) -> CompiledGraph:
        """Build the LangGraph workflow"""
```

### Agent Interface

```python
class BaseAgent:
    async def run(self, state: State) -> Dict[str, Any]:
        """Execute agent logic"""
        
    def bind_tools(self, tools: List[BaseTool]) -> None:
        """Bind MCP tools to agent"""
```

## 🐛 Troubleshooting

### Common Issues

1. **MCP Gateway Connection Failed**
   ```bash
   # Check if gateway is running
   curl http://localhost:8808/health
   
   # Restart gateway
   cd gateway && python -m mcp_gateway.server
   ```

2. **Tool Loading Errors**
   ```bash
   # Verify MCP tools are available
   python -c "from src.react_agent.tools import _load_tools; print(len(_load_tools()))"
   ```

3. **Authentication Issues**
   ```bash
   # Check environment variables
   python -c "import os; print('ANTHROPIC_API_KEY' in os.environ)"
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the workflow framework
- [Model Context Protocol](https://github.com/modelcontextprotocol) for tool integration
- [Anthropic](https://anthropic.com) for Claude API
- [LangChain](https://langchain.com) for the foundation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/cuoiKyNLP3/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/cuoiKyNLP3/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/cuoiKyNLP3/wiki)

---

Made with ❤️ for the AI community
