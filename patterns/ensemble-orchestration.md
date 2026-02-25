# Pattern: Ensemble Agent Orchestration

## Multi-Agent Ensemble Design

The ensemble system combines three specialized agents — each using a different approach to the same task — and a meta-agent that produces a weighted combination of their outputs.

### Agent Hierarchy

```
Scanner Agent (data collection)
    │
    ├──→ Specialist Agent (domain heuristics)
    ├──→ Frontier Agent (LLM + RAG)
    └──→ ML Agent (Random Forest)
            │
            ▼
    Ensemble Agent (Linear Regression combiner)
            │
            ▼
       Final Output
```

### Base Agent Contract

All agents inherit from a common base with standardized interfaces and colored logging for observability.

```python
class Agent:
    """Base agent with consistent interface and logging."""

    name: str
    color: str  # Terminal color for distinguishing agent output
    model_name: str

    def __init__(self, collection: dict):
        """Initialize with shared data collection."""
        self.collection = collection
        self.log(f"Initialized with {len(collection)} items")

    def log(self, message: str):
        """Color-coded logging for multi-agent observability."""
        print(f"{self.color}[{self.name}] {message}{RESET}")
```

### Specialist Agent (Domain Knowledge)

Applies business rules and domain-specific heuristics. No LLM dependency — pure deterministic logic.

```python
class SpecialistAgent(Agent):
    """Domain expert using business rules."""

    def estimate(self, item: dict) -> float:
        """Apply domain heuristics to produce an estimate."""
        base_value = self.lookup_category_baseline(item["category"])
        adjustments = self.apply_rules(item)  # Condition, market, rarity
        return base_value * adjustments

    def apply_rules(self, item: dict) -> float:
        """Business rules engine — deterministic, explainable."""
        factor = 1.0
        if item["condition"] == "new":
            factor *= 1.2
        if item["rarity"] == "limited":
            factor *= 1.5
        # ... additional domain rules
        return factor
```

### Frontier Agent (LLM + RAG)

Uses a frontier LLM augmented with retrieval-augmented generation for contextual reasoning.

```python
class FrontierAgent(Agent):
    """LLM reasoning with RAG context."""

    def __init__(self, collection: dict):
        super().__init__(collection)
        # Build vector store from domain knowledge
        self.vectorstore = Chroma.from_documents(
            documents=self.load_knowledge_base(),
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    def estimate(self, item: dict) -> float:
        """LLM reasoning with RAG context retrieval."""
        # Retrieve relevant context from knowledge base
        context_docs = self.vectorstore.similarity_search(
            item["description"], k=5
        )
        context = "\n".join([doc.page_content for doc in context_docs])

        # LLM produces estimate with context
        prompt = f"""Given the following context:
        {context}

        Estimate the value for: {item['description']}
        Return only a number."""

        response = self.llm.invoke(prompt)
        return parse_number(response)
```

### ML Agent (Random Forest)

Trained machine learning model for structured prediction.

```python
class RandomForestAgent(Agent):
    """ML-based estimation using trained Random Forest."""

    def __init__(self, collection: dict):
        super().__init__(collection)
        self.model = self.train_model()

    def train_model(self) -> RandomForestRegressor:
        """Train on historical data."""
        X = self.extract_features(self.collection)
        y = self.extract_targets(self.collection)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        return model

    def estimate(self, item: dict) -> float:
        """Predict using trained model."""
        features = self.extract_features([item])
        return self.model.predict(features)[0]
```

### Ensemble Agent (Combiner)

Combines all agent outputs using a learned weighting.

```python
class EnsembleAgent(Agent):
    """Combines agent estimates via Linear Regression."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.combiner = LinearRegression()

    def train_combiner(self, items: list[dict], ground_truth: list[float]):
        """Learn optimal weights for each agent's estimates."""
        # Collect estimates from all agents
        estimates = []
        for item in items:
            row = [agent.estimate(item) for agent in self.agents]
            row.append(min(row))  # Min aggregate
            row.append(max(row))  # Max aggregate
            estimates.append(row)

        X = np.array(estimates)
        y = np.array(ground_truth)
        self.combiner.fit(X, y)

    def ensemble_estimate(self, item: dict) -> float:
        """Produce weighted combination of all agents."""
        individual = [agent.estimate(item) for agent in self.agents]
        features = individual + [min(individual), max(individual)]
        return self.combiner.predict([features])[0]
```

### Pipeline Execution

```python
# Sequential pipeline with shared state
scanner = ScannerAgent()               # Collect input data
opportunities = scanner.scan()          # Returns list of items

specialist = SpecialistAgent(opportunities)
frontier = FrontierAgent(opportunities)
ml_agent = RandomForestAgent(opportunities)

ensemble = EnsembleAgent([specialist, frontier, ml_agent])
ensemble.train_combiner(training_data, training_labels)

# Produce final results
for item in opportunities:
    result = ensemble.ensemble_estimate(item)
    confidence = calculate_agent_agreement(
        specialist.estimate(item),
        frontier.estimate(item),
        ml_agent.estimate(item)
    )
    print(f"{item['name']}: {result:.2f} (confidence: {confidence:.0%})")
```

## Key Design Properties

1. **Agent independence**: Each agent produces results without knowing about the others. No agent can influence another's output.
2. **Failure isolation**: If one agent errors, the ensemble can still produce a result from the remaining agents (with degraded confidence).
3. **Explainability**: Individual agent estimates are preserved alongside the ensemble result, enabling "why did the system predict X?" analysis.
4. **Extensibility**: Adding a new agent requires only implementing the `estimate()` interface and adding it to the ensemble's agent list.
