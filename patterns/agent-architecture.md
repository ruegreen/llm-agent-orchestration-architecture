# Pattern: Agent Architecture

## ReAct Agent with LangGraph

The core agent implements the ReAct (Reasoning + Acting) pattern as a LangGraph `StateGraph`. The agent alternates between reasoning (LLM decides what to do) and acting (executing tools), maintaining full conversation state across iterations.

### State Definition

```python
# Custom state preserving conversation context across reasoning steps
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Full conversation history
    iteration_count: int                      # Reasoning loop counter
    final_answer: Optional[str]               # Terminal output

# add_messages reducer appends rather than replaces — critical for
# preserving tool call results in the conversation history
```

### Graph Construction

```python
# Two-node graph: agent reasons, tools execute, loop continues
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)    # LLM reasoning step
graph.add_node("tools", tools_node)    # Tool execution step

graph.set_entry_point("agent")

# Conditional routing: if the agent wants to call tools, route to tools node
# Otherwise, end the conversation
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})

# After tools execute, always return to agent for next reasoning step
graph.add_edge("tools", "agent")

agent = graph.compile(checkpointer=MemorySaver())
```

### Conversation Memory

```python
# Thread-based isolation — each conversation has independent memory
config = {"configurable": {"thread_id": thread_id}}

# Synchronous execution
result = agent.invoke({"messages": [user_message]}, config)

# Streaming execution (token-by-token)
for chunk in agent.stream({"messages": [user_message]}, config):
    yield chunk

# Retrieve conversation history for a thread
history = agent.get_state(config).values.get("messages", [])
```

### Iteration Safety

```python
def should_continue(state: AgentState) -> str:
    """Prevent runaway reasoning loops."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"
```

## Dual Agent Pattern

The API server maintains two agent instances for different latency profiles:

```python
# Full agent with tool access — higher latency, richer capability
agents["default"] = create_agent(tools=all_tools, model=primary_model)

# Simple agent without tools — lower latency for straightforward queries
agents["simple"] = create_agent(tools=[], model=fast_model)
```

The client chooses which agent to invoke:
- `POST /agent` → full tool-use agent (reasoning + enterprise tools)
- `POST /chat` → simple agent (direct LLM response, no tools)
