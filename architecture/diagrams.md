# Architecture Diagrams — LLM Agent Orchestration

## 1. High-Level System Architecture

The full system spans four layers: client interfaces, agent runtime, local LLM inference, and enterprise tool integration.

```mermaid
graph TB
    subgraph Clients["Client Interfaces"]
        CLI["CLI Interface"]
        API["REST API (FastAPI)"]
        WEBUI["Open WebUI"]
    end

    subgraph AgentRuntime["Agent Runtime (LangGraph)"]
        ORCH["Orchestrator (ReAct Loop)"]
        STATE["State Manager (MemorySaver)"]
        TOOLS["Tool Registry"]
    end

    subgraph LLMLayer["Local LLM Layer"]
        OLLAMA["Ollama Server"]
        MODELS["Model Store (3B → 70B)"]
        GPU["GPU Hardware (CUDA / Metal)"]
    end

    subgraph EnterpriseMCP["Enterprise Integration (MCP)"]
        MCPC["MCP Client (HTTP/SSE)"]
        SN["ITSM Platform"]
        CRM2["CRM Platform"]
        EHR2["EHR Systems"]
    end

    subgraph BuiltIn["Built-in Tools"]
        SEARCH["Web Search"]
        FILES["File Operations"]
        CALC["Calculator"]
        SHELL["Shell Execution"]
    end

    CLI & API & WEBUI --> ORCH
    ORCH --> STATE
    ORCH --> TOOLS
    TOOLS --> BuiltIn
    TOOLS --> MCPC
    MCPC --> SN & CRM2 & EHR2
    ORCH --> OLLAMA
    OLLAMA --> MODELS
    MODELS --> GPU
```

## 2. ReAct Agent Loop

The production agent follows the ReAct (Reasoning + Acting) pattern implemented as a LangGraph StateGraph with two nodes and a conditional edge.

```mermaid
graph LR
    START((Start)) --> AGENT["Agent Node\n(LLM Reasoning)"]
    AGENT --> DECIDE{Tool Call\nNeeded?}
    DECIDE -->|Yes| TOOLS["Tool Node\n(Execute Tools)"]
    TOOLS --> COUNT{Iteration\nLimit?}
    COUNT -->|Under Limit| AGENT
    COUNT -->|Over Limit| END((End))
    DECIDE -->|No| END
```

**State management**: The `AgentState` TypedDict carries `messages` (full conversation with tool results), `iteration_count` (loop counter), and `final_answer` across all nodes. The `add_messages` reducer appends new messages rather than replacing them.

**Iteration safety**: A configurable maximum (default: 10) prevents runaway reasoning loops. If the agent exceeds the limit, the loop terminates and returns the last available response.

## 3. Multi-Agent Ensemble Flow

The benchmarking system orchestrates multiple specialized agents to produce a weighted ensemble result.

```mermaid
graph TB
    INPUT["Input Task"] --> SCANNER["Scanner Agent\n(Data Collection)"]
    SCANNER --> SPEC["Specialist Agent\n(Domain Knowledge)"]
    SCANNER --> FRONTIER["Frontier Agent\n(LLM + RAG)"]
    SCANNER --> ML["ML Agent\n(Random Forest)"]
    SPEC & FRONTIER & ML --> ENSEMBLE["Ensemble Agent\n(Linear Regression)"]
    ENSEMBLE --> OUTPUT["Final Output\n(Weighted Result)"]

    subgraph RAG["RAG Pipeline"]
        EMBED["Embedding Model\n(MiniLM-L6-v2)"]
        VECTOR["ChromaDB\nVector Store"]
        FRONTIER -.-> EMBED --> VECTOR
        VECTOR -.-> FRONTIER
    end
```

**Agent roles**:
- **Scanner**: Collects input data from external sources (RSS feeds, web scraping)
- **Specialist**: Applies domain-specific heuristics and knowledge rules
- **Frontier**: Uses LLM reasoning augmented with RAG (ChromaDB vector search over domain knowledge base)
- **ML Agent**: Runs a trained Random Forest model for structured prediction
- **Ensemble**: Combines all three outputs using Linear Regression to produce a weighted final result

## 4. MCP Tool Integration Flow

Enterprise systems are accessed through the Model Context Protocol (MCP) via an HTTP/SSE client that handles tool discovery, authentication, and execution.

```mermaid
sequenceDiagram
    participant Agent as LangGraph Agent
    participant Registry as Tool Registry
    participant MCP as MCP Client
    participant Server as MCP Server
    participant API as Enterprise API

    Note over Agent,API: Startup — Tool Discovery
    Agent->>Registry: Initialize tools
    Registry->>MCP: Connect to MCP servers
    MCP->>Server: POST /mcp (initialize)
    Server-->>MCP: Session ID + capabilities
    MCP->>Server: POST /mcp (tools/list)
    Server-->>MCP: Tool definitions + schemas
    MCP-->>Registry: Convert to LangChain StructuredTools
    Registry-->>Agent: Tools available

    Note over Agent,API: Runtime — Tool Execution
    Agent->>Registry: Call tool (name, args)
    Registry->>MCP: Execute via MCP client
    MCP->>Server: POST /mcp (tools/call) + session header
    Server->>API: Platform-specific API call
    API-->>Server: API response
    Server-->>MCP: Tool result
    MCP-->>Registry: Parsed result
    Registry-->>Agent: Tool output → next reasoning step
```

## 5. Deployment Topology

Three deployment modes from a single codebase, using environment-specific docker-compose overrides.

```mermaid
graph TB
    subgraph Local["Local Development"]
        L_OLLAMA["Ollama (native)\nApple Metal"]
        L_AGENT["Python venv\nFastAPI + CLI"]
        L_OLLAMA <--> L_AGENT
    end

    subgraph Container["Container (Podman/Docker)"]
        C_OLLAMA["Ollama Container\nCPU inference"]
        C_AGENT["Agent API Container\nFastAPI"]
        C_WEBUI["Open WebUI Container"]
        C_NET["llm-network (bridge)"]
        C_OLLAMA & C_AGENT & C_WEBUI --- C_NET
    end

    subgraph GPU["GPU / AI Pod (Kubernetes)"]
        G_OLLAMA["Ollama Pod\nNVIDIA CDI\nFlash Attention"]
        G_AGENT["Agent API Pod\nHPA Autoscaling"]
        G_WEBUI["Open WebUI Pod"]
        G_PVC["PVC Storage\n(Model Weights)"]
        G_OLLAMA --> G_PVC
        G_AGENT <--> G_OLLAMA
        G_WEBUI <--> G_OLLAMA
    end

    style Local fill:#e8f5e9
    style Container fill:#e3f2fd
    style GPU fill:#fff3e0
```

## 6. Model Size vs Capability Matrix

| Model | VRAM | Latency | Tool-Use Quality | Reasoning | Recommended For |
|---|---|---|---|---|---|
| 3B | 2 GB | <1s | Adequate | Basic | Development, testing |
| 7B | 5 GB | 1-2s | Good | Moderate | Production agent (standard) |
| 14B | 9 GB | 2-4s | Strong | Good | Production agent (complex) |
| 32B | 20 GB | 4-8s | Excellent | Strong | Batch processing, analysis |
| 70B | 40 GB | 8-15s | Excellent | Excellent | Maximum quality (offline) |
