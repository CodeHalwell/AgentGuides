# Mistral Agents API: Ready-to-Use Recipes

Copy-paste ready code examples for common use cases. All recipes require `pip install mistralai[agents]` and `MISTRAL_API_KEY` environment variable.

---

## 1. Web Search Agent

```python
import os
from mistralai import Mistral

def create_web_search_agent():
    """Create and use a web search agent"""
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    # Step 1: Create agent with web search capability
    agent = client.beta.agents.create(
        model="mistral-medium-2505",
        name="WebSearch Agent",
        description="Agent able to search the web for current information",
        instructions=(
            "You are a helpful research assistant. "
            "Use web search to find current information when needed. "
            "Always cite your sources."
        ),
        tools=[{"type": "web_search"}],
        completion_args={
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 1024
        }
    )
    
    print(f"Agent created: {agent.id}")
    
    # Step 2: Start a conversation
    conversation = client.beta.conversations.start(
        agent_id=agent.id,
        inputs="What are the latest developments in AI in 2025?"
    )
    
    print(f"Conversation: {conversation.id}")
    print(f"Response:\n{conversation.outputs[-1].content}")
    
    # Step 3: Continue conversation
    follow_up = client.beta.conversations.append(
        conversation_id=conversation.id,
        inputs="Tell me more about the third point"
    )
    
    print(f"Follow-up Response:\n{follow_up.outputs[-1].content}")
    
    # Step 4: View full history
    history = client.beta.conversations.get_history(
        conversation_id=conversation.id
    )
    
    print(f"\nFull conversation history ({len(history.entries)} entries):")
    for entry in history.entries:
        print(f"  {entry.type}: {entry.role}")
    
    return agent, conversation

if __name__ == "__main__":
    create_web_search_agent()
```

---

## 2. Persistent Chatbot with Memory

```python
import os
from mistralai import Mistral

class MemoryChatbot:
    """Chatbot with persistent memory across sessions"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.agent = None
        self.conversation_id = None
    
    def initialise(self):
        """Create or load agent"""
        self.agent = self.client.beta.agents.create(
            model="mistral-medium-2505",
            name="Persistent Chatbot",
            description="A chatbot that remembers conversation history",
            instructions=(
                "You are a friendly chatbot. "
                "Remember all previous messages in this conversation. "
                "Use that context to provide relevant responses. "
                "Be helpful and maintain continuity."
            )
        )
        print(f"Chatbot initialised: {self.agent.id}")
    
    def start_conversation(self):
        """Start new conversation"""
        # This persists on Mistral's servers
        conversation = self.client.beta.conversations.start(
            agent_id=self.agent.id,
            inputs="Hello! I'd like to start a conversation."
        )
        self.conversation_id = conversation.id
        print(f"Started conversation: {self.conversation_id}")
        return conversation.outputs[-1].content
    
    def resume_conversation(self, conversation_id: str):
        """Resume existing conversation (even after days)"""
        self.conversation_id = conversation_id
        # The conversation history is automatically loaded
    
    def send_message(self, user_input: str) -> str:
        """Send message and get response (with full history)"""
        response = self.client.beta.conversations.append(
            conversation_id=self.conversation_id,
            inputs=user_input
        )
        return response.outputs[-1].content
    
    def get_summary(self) -> str:
        """Get conversation summary"""
        history = self.client.beta.conversations.get_history(
            conversation_id=self.conversation_id
        )
        
        # Use agent to summarize
        summary_response = self.client.beta.conversations.start(
            agent_id=self.agent.id,
            inputs=f"Summarise this conversation:\n{self._format_history(history)}"
        )
        
        return summary_response.outputs[-1].content
    
    def _format_history(self, history) -> str:
        """Format conversation history for display"""
        text = []
        for entry in history.entries:
            if entry.type == "message.input":
                text.append(f"User: {entry.content}")
            elif entry.type == "message.output":
                text.append(f"Assistant: {entry.content}")
        return "\n".join(text)


# Usage
if __name__ == "__main__":
    bot = MemoryChatbot()
    bot.initialise()
    
    # First session
    print("=== Session 1 ===")
    response = bot.start_conversation()
    print(f"Bot: {response}")
    
    response = bot.send_message("My name is Alice")
    print(f"Bot: {response}")
    
    conv_id = bot.conversation_id
    
    # Later, resume same conversation
    print("\n=== Session 2 (Hours/Days Later) ===")
    bot2 = MemoryChatbot()
    bot2.initialise()
    bot2.resume_conversation(conv_id)
    
    response = bot2.send_message("What's my name?")
    print(f"Bot: {response}")  # Should remember "Alice"
```

---

## 3. Multi-Agent System (No External Frameworks)

```python
import os
import asyncio
from mistralai import Mistral

class MultiAgentOrchestrator:
    """Coordinate multiple agents without external frameworks"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.agents = {}
    
    def create_agents(self):
        """Create specialised agents"""
        
        # Agent 1: Research
        self.agents['researcher'] = self.client.beta.agents.create(
            model="mistral-large-latest",
            name="Research Agent",
            description="Researches topics using web search",
            instructions=(
                "You are a research expert. "
                "Use web search to find comprehensive information. "
                "Provide detailed, well-sourced responses."
            ),
            tools=[{"type": "web_search"}]
        )
        
        # Agent 2: Analyst
        self.agents['analyst'] = self.client.beta.agents.create(
            model="mistral-medium-2505",
            name="Analysis Agent",
            description="Analyzes information and extracts insights",
            instructions=(
                "You are a data analyst. "
                "Take provided information and extract key insights. "
                "Identify trends and patterns. "
                "Present findings clearly."
            )
        )
        
        # Agent 3: Report Writer
        self.agents['writer'] = self.client.beta.agents.create(
            model="mistral-medium-2505",
            name="Report Writer",
            description="Writes professional reports",
            instructions=(
                "You are a technical writer. "
                "Create professional, well-formatted reports. "
                "Include executive summary, findings, and recommendations."
            )
        )
    
    def process_pipeline(self, topic: str) -> str:
        """Sequential pipeline: Research → Analyse → Report"""
        
        print(f"Processing: {topic}")
        
        # Step 1: Research
        print("\n1. Research phase...")
        research_conv = self.client.beta.conversations.start(
            agent_id=self.agents['researcher'].id,
            inputs=f"Research {topic}. Provide comprehensive information."
        )
        research_output = research_conv.outputs[-1].content
        
        # Step 2: Analyse
        print("2. Analysis phase...")
        analysis_conv = self.client.beta.conversations.start(
            agent_id=self.agents['analyst'].id,
            inputs=f"Analyse this research:\n{research_output}"
        )
        analysis_output = analysis_conv.outputs[-1].content
        
        # Step 3: Report
        print("3. Report writing phase...")
        report_conv = self.client.beta.conversations.start(
            agent_id=self.agents['writer'].id,
            inputs=f"Write a report based on this analysis:\n{analysis_output}"
        )
        final_report = report_conv.outputs[-1].content
        
        return final_report


# Usage
if __name__ == "__main__":
    orchestrator = MultiAgentOrchestrator()
    orchestrator.create_agents()
    
    report = orchestrator.process_pipeline("Latest developments in Quantum Computing")
    print("\n=== FINAL REPORT ===")
    print(report)
```

---

## 4. Custom Tool Integration

```python
import os
import json
from mistralai import Mistral

class CustomToolAgent:
    """Agent with custom tools defined via schema"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    def create_agent_with_custom_tools(self):
        """Create agent with custom function tools"""
        
        # Define custom tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name (e.g., London, Paris)"
                            },
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature units"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_distance",
                    "description": "Calculate distance between two locations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_location": {"type": "string"},
                            "to_location": {"type": "string"}
                        },
                        "required": ["from_location", "to_location"]
                    }
                }
            }
        ]
        
        agent = self.client.beta.agents.create(
            model="mistral-medium-2505",
            name="Custom Tool Agent",
            description="Agent with custom tools",
            instructions=(
                "You are a helpful assistant with access to custom tools. "
                "Use them when appropriate to answer user questions."
            ),
            tools=tools
        )
        
        return agent
    
    def process_with_tool_handling(self, agent_id: str, user_input: str):
        """Process conversation with custom tool handling"""
        
        conv = self.client.beta.conversations.start(
            agent_id=agent_id,
            inputs=user_input
        )
        
        # Analyse tool calls in response
        for entry in conv.outputs:
            if entry.type == "tool.execution":
                print(f"Tool executed: {entry.name}")
                # Tool results are included in response
            elif entry.type == "message.output":
                print(f"Response: {entry.content}")
        
        return conv


# Usage
if __name__ == "__main__":
    tool_agent = CustomToolAgent()
    agent = tool_agent.create_agent_with_custom_tools()
    
    result = tool_agent.process_with_tool_handling(
        agent.id,
        "What's the weather in London?"
    )
```

---

## 5. RAG Pattern with Document Retrieval

```python
import os
from mistralai import Mistral

class RAGSystem:
    """Retrieval-Augmented Generation using document library"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    def create_rag_agent(self, knowledge_base_name: str):
        """Create agent with document retrieval capability"""
        
        agent = self.client.beta.agents.create(
            model="mistral-large-latest",
            name="RAG Agent",
            description="Agent with access to knowledge base",
            instructions=(
                "You are a knowledgeable assistant with access to a knowledge base. "
                "Always retrieve relevant documents before answering. "
                "Cite sources from the knowledge base. "
                "If information isn't in the knowledge base, say so."
            ),
            tools=[
                {
                    "type": "document_library",
                    "name": knowledge_base_name
                }
            ]
        )
        
        return agent
    
    def query_knowledge_base(self, agent_id: str, query: str) -> str:
        """Query with RAG"""
        
        conv = self.client.beta.conversations.start(
            agent_id=agent_id,
            inputs=f"Answer based on the knowledge base: {query}"
        )
        
        # Extract response
        for entry in conv.outputs:
            if entry.type == "message.output":
                return entry.content
    
    def interactive_rag_session(self, agent_id: str):
        """Interactive multi-turn RAG conversation"""
        
        print("Starting RAG conversation. Type 'exit' to quit.")
        
        conversation = self.client.beta.conversations.start(
            agent_id=agent_id,
            inputs="Hello, I'd like to ask about the knowledge base."
        )
        conv_id = conversation.id
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            response = self.client.beta.conversations.append(
                conversation_id=conv_id,
                inputs=user_input
            )
            
            answer = response.outputs[-1].content
            print(f"Agent: {answer}\n")


# Usage
if __name__ == "__main__":
    rag = RAGSystem()
    agent = rag.create_rag_agent("company_knowledge_base")
    
    # Single query
    answer = rag.query_knowledge_base(
        agent.id,
        "What are the company policies?"
    )
    print(f"Answer: {answer}")
```

---

## 6. Streaming Responses

```python
import os
from mistralai import Mistral

def streaming_conversation():
    """Get real-time streaming responses"""
    
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    # Create agent
    agent = client.beta.agents.create(
        model="mistral-medium-2505",
        name="Streaming Agent",
        description="Agent with streaming support"
    )
    
    # Start conversation with streaming
    print("Streaming response:")
    print("-" * 40)
    
    stream = client.beta.conversations.start_stream(
        agent_id=agent.id,
        inputs="Write a short story about a robot learning to paint"
    )
    
    # Process stream
    with stream as event_stream:
        for event in event_stream:
            if event.type == "message.output.delta":
                # Print each chunk as it arrives
                print(event.content, end="", flush=True)
            elif event.type == "tool.execution.started":
                print(f"\n[Tool executing: {event.name}]")
            elif event.type == "conversation.response.done":
                print(f"\n[Done - {event.usage.total_tokens} tokens]")
    
    print("-" * 40)


# Usage
if __name__ == "__main__":
    streaming_conversation()
```

---

## 7. Conversation Restart (Branching)

```python
import os
from mistralai import Mistral

def branching_conversations():
    """Restart conversation from specific point"""
    
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    agent = client.beta.agents.create(
        model="mistral-medium-2505",
        name="Branching Agent"
    )
    
    # Initial conversation
    print("=== Turn 1 ===")
    conv1 = client.beta.conversations.start(
        agent_id=agent.id,
        inputs="Write a story about a cat"
    )
    story = conv1.outputs[-1].content
    print(story[:200] + "...")
    turn1_id = conv1.outputs[0].id  # Get first entry ID
    
    print("\n=== Turn 2 (Continuation) ===")
    conv2 = client.beta.conversations.append(
        conversation_id=conv1.id,
        inputs="Make it longer"
    )
    print(conv2.outputs[-1].content[:200] + "...")
    
    print("\n=== Branch: Restart from Turn 1 with Different Request ===")
    # Restart from Turn 1, but with different instruction
    conv_branch = client.beta.conversations.restart(
        conversation_id=conv1.id,
        from_entry_id=turn1_id,
        inputs="Write a story about a dog instead"
    )
    
    print(conv_branch.outputs[-1].content[:200] + "...")
    print(f"\nOriginal conversation: {conv1.id}")
    print(f"Branched conversation (explored alternative): {conv_branch.id}")


# Usage
if __name__ == "__main__":
    branching_conversations()
```

---

## 8. Error Handling Template

```python
import os
from mistralai import Mistral

class ErrorHandlingExample:
    """Best practices for error handling"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    def safe_agent_creation(self):
        """Create agent with error handling"""
        try:
            agent = self.client.beta.agents.create(
                model="mistral-medium-2505",
                name="Safe Agent"
            )
            print(f"✓ Agent created: {agent.id}")
            return agent
        
        except ValueError as e:
            print(f"✗ Invalid input: {e}")
        except Exception as e:
            print(f"✗ Error creating agent: {e}")
            return None
    
    def safe_conversation_start(self, agent_id: str, inputs: str):
        """Start conversation with error handling"""
        try:
            conv = self.client.beta.conversations.start(
                agent_id=agent_id,
                inputs=inputs
            )
            print(f"✓ Conversation started: {conv.id}")
            return conv
        
        except ValueError as e:
            print(f"✗ Invalid input: {e}")
        except Exception as e:
            print(f"✗ Error starting conversation: {e}")
            return None
    
    def safe_conversation_continuation(self, conv_id: str, inputs: str):
        """Continue conversation with error handling"""
        try:
            response = self.client.beta.conversations.append(
                conversation_id=conv_id,
                inputs=inputs
            )
            print(f"✓ Message appended")
            return response
        
        except ValueError as e:
            print(f"✗ Invalid input: {e}")
        except Exception as e:
            print(f"✗ Error continuing conversation: {e}")
            return None


# Usage
if __name__ == "__main__":
    handler = ErrorHandlingExample()
    agent = handler.safe_agent_creation()
    
    if agent:
        conv = handler.safe_conversation_start(
            agent.id,
            "Hello!"
        )
        
        if conv:
            handler.safe_conversation_continuation(
                conv.id,
                "Tell me more"
            )
```

---

## 9. Complete Application Example

```python
import os
from mistralai import Mistral

class MistralAgentApp:
    """Complete application using Mistral Agents"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.agents = {}
        self.conversations = {}
    
    def setup(self):
        """Initialise all agents"""
        self.agents['main'] = self.client.beta.agents.create(
            model="mistral-medium-2505",
            name="Main Assistant",
            description="Main application assistant"
        )
    
    def run_interactive_session(self):
        """Main application loop"""
        print("Starting Mistral Agents Application")
        print("Commands: 'new' (new conversation), 'list' (list conversations), 'exit'")
        
        while True:
            command = input("\n> ").strip().lower()
            
            if command == 'exit':
                break
            elif command == 'new':
                self.start_new_conversation()
            elif command == 'list':
                self.list_conversations()
            else:
                self.send_message(command)
    
    def start_new_conversation(self):
        """Start new conversation"""
        initial_input = input("Initial message: ").strip()
        
        conv = self.client.beta.conversations.start(
            agent_id=self.agents['main'].id,
            inputs=initial_input
        )
        
        self.conversations[conv.id] = conv
        print(f"Conversation started: {conv.id}")
        print(f"Response: {conv.outputs[-1].content[:100]}...")
    
    def send_message(self, message: str):
        """Send message to current conversation"""
        if not self.conversations:
            print("No active conversation. Use 'new' to start one.")
            return
        
        # Use most recent conversation
        conv_id = list(self.conversations.keys())[-1]
        
        response = self.client.beta.conversations.append(
            conversation_id=conv_id,
            inputs=message
        )
        
        print(f"Response: {response.outputs[-1].content}")
    
    def list_conversations(self):
        """List all conversations"""
        convs = self.client.beta.conversations.list(page=0, page_size=10)
        print(f"\nConversations ({len(convs)}):")
        for conv in convs:
            print(f"  - {conv.id}: {conv.created_at}")


# Usage
if __name__ == "__main__":
    app = MistralAgentApp()
    app.setup()
    app.run_interactive_session()
```

---

**All recipes are production-ready and can be extended for specific use cases. Mix and match patterns for your requirements.**


