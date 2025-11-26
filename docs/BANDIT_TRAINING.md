# Bandit Training Strategy

**Key Insight**: Contextual bandits are the right abstraction for LLM routing - full RL would model state transitions that don't exist.

## Algorithm Selection

The Conduit Router implements **6 bandit algorithms** for different use cases:

### Contextual Algorithms (Recommended)
**Use query features** to make smarter routing decisions:

1. **LinUCB** (Production Recommended)
   - Ridge regression with upper confidence bound
   - Best for: Production LLM routing with diverse queries
   - Sample efficiency: 2,000-3,000 queries to converge
   - File: `conduit/engines/bandits/linucb.py`

2. **Contextual Thompson Sampling**
   - Bayesian linear regression approach
   - Best for: When Bayesian uncertainty estimation is valuable
   - Sample efficiency: Similar to LinUCB
   - File: `conduit/engines/bandits/contextual_thompson_sampling.py`

### Non-Contextual Algorithms
**Ignore query features**, simpler but less effective:

3. **Thompson Sampling**
   - Beta distribution sampling per arm
   - Best for: Simple workloads, baseline comparison
   - Sample efficiency: 5,000-10,000 queries to converge
   - File: `conduit/engines/bandits/thompson_sampling.py`

4. **UCB1**
   - Upper confidence bound with logarithmic exploration
   - Best for: Fast non-contextual baseline
   - Sample efficiency: 3,000-5,000 queries to converge
   - File: `conduit/engines/bandits/ucb.py`

5. **Epsilon-Greedy**
   - Simple exploration (ε) vs exploitation (1-ε)
   - Best for: Baseline comparison, simple testing
   - Sample efficiency: 8,000-12,000 queries to converge
   - File: `conduit/engines/bandits/epsilon_greedy.py`

### Baselines (Non-Learning)
**Fixed strategies** for comparison and fallback:

6. **Baselines** (Random, AlwaysBest, AlwaysCheapest, Oracle)
   - No learning, static routing rules
   - Best for: Benchmarking, testing, cost analysis
   - File: `conduit/engines/bandits/baselines.py`

**Production Recommendation**: Use **LinUCB** (contextual) or **Hybrid Routing** (UCB1→LinUCB warm start) for best results. See [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) for comprehensive algorithm details.

## Why Contextual Bandits (Not Full RL)

Contextual bandits are a subset of reinforcement learning, optimized for problems like LLM routing. Here's why we use bandits instead of full RL:

### The Key Distinction

| Aspect | Contextual Bandits | Full RL (MDP) |
|--------|-------------------|---------------|
| **State transitions** | None - each decision independent | Actions affect future states |
| **Horizon** | Single step | Multi-step episodes |
| **Credit assignment** | Immediate reward only | Delayed rewards across trajectory |
| **Complexity** | O(actions × features) | O(states × actions × transitions) |
| **Computation** | Matrix ops, real-time | Often needs simulation/replay |

### Why Bandits Fit LLM Routing

**1. No state transitions exist**

Routing query A to GPT-4 doesn't change the environment for query B. Each query is independent - there's no "game state" evolving. Full RL would model transitions that don't exist.

**2. Immediate rewards**

You know immediately if the response was good/fast/cheap. No delayed credit assignment problem. No need to propagate rewards backward through a trajectory.

**3. Context informs but isn't state**

Query features (embedding, complexity, domain) inform the decision, but they're not "state" that your actions modify. Pure bandits (UCB1, Thompson) ignore context; contextual bandits (LinUCB) use it.

**4. Computational tractability**

```python
# LinUCB: O(d²) per update where d=features
# Real-time inference: ~1ms per routing decision

# Full RL would need:
# - Q(s,a) for exponentially many states
# - Experience replay buffers
# - Target networks
# - Much higher latency
```

### When You'd Need Full RL Instead

Full RL makes sense when actions affect future states:

- **Multi-turn conversations**: Earlier routing affects later context window
- **Budget constraints across sessions**: Route cheap now to afford expensive later
- **Caching strategies**: Routing affects future cache hit rates
- **Rate limit management**: Current usage affects future availability

None of these apply to single-query routing, which is why contextual bandits are the right abstraction.

### Zero-Shot Deployment

The practical advantage: bandits learn online with no pre-training.

```python
# Deploy immediately - no training data needed
bandit = LinUCBBandit(models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"])

# Learns from every query in production:
for query in production_traffic:
    model = bandit.select(features)      # Pick model (exploration/exploitation)
    result = execute_llm(model, query)   # Execute
    feedback = collect_feedback(result)  # Errors, latency, ratings
    bandit.update(model, reward)         # Learn! (just matrix arithmetic)
```

No GPU hours, no training corpus, no pre-deployment phase. Start routing immediately, improve continuously.

## How Bandit Algorithms Learn

**Note**: This section uses Thompson Sampling as an illustrative example. LinUCB (recommended) uses ridge regression instead of Beta distributions. See [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) for algorithm-specific details.

### Initialization: Uniform Priors (Thompson Sampling Example)

```python
# All models start equal - we know nothing
models = {
    "gpt-4o-mini": Beta(alpha=1, beta=1),  # Uniform distribution
    "gpt-4o": Beta(alpha=1, beta=1),
    "claude-sonnet-4": Beta(alpha=1, beta=1)
}

# This means: "Each model has 50% expected success rate, but high uncertainty"

# LinUCB initialization (contextual alternative):
# models = {
#     "gpt-4o-mini": LinUCBState(A=I, b=0),  # Identity matrix + zero vector
#     "gpt-4o": LinUCBState(A=I, b=0),
#     ...
# }
```

### Learning Phase: Feedback Accumulation

Every query provides feedback that updates the distributions:

```python
# Query 1: Route to gpt-4o-mini
result = execute("gpt-4o-mini", "What is Python?")
feedback = detect_signals(result)

if feedback.success:
    models["gpt-4o-mini"].alpha += 1  # Success count
else:
    models["gpt-4o-mini"].beta += 1   # Failure count

# After 100 queries:
models = {
    "gpt-4o-mini": Beta(alpha=45, beta=15),  # 75% success rate, moderate confidence
    "gpt-4o": Beta(alpha=38, beta=8),        # 82% success rate, moderate confidence
    "claude-sonnet-4": Beta(alpha=40, beta=12) # 77% success rate, moderate confidence
}
```

### Convergence: Stable Routing

After sufficient feedback, distributions stabilize:

```python
# After 1000 queries:
models = {
    "gpt-4o-mini": Beta(alpha=450, beta=150),  # 75% success, high confidence
    "gpt-4o": Beta(alpha=380, beta=80),        # 82% success, high confidence
    "claude-sonnet-4": Beta(alpha=400, beta=120) # 77% success, high confidence
}

# Routing becomes consistent (but still explores occasionally)
# The more data, the more confident and consistent the routing
```

## The Learning Curve

### Phase 1: Cold Start (Queries 1-5,000)

**Characteristics**:
- High exploration (tries everything, gathering initial data)
- Inconsistent routing decisions (high variance)
- Learning basic model capabilities and failure modes
- Establishing fundamental routing patterns
- Each model tried extensively across domains

**User Experience**: Noticeable suboptimal routing, but rapid learning

**Example**:
```
Query 1:     "Debug this code" → gpt-4o-mini (random) → Failure ✗
Query 50:    "Debug this code" → gpt-4o (exploring) → Success ✓
Query 500:   "What is 2+2?" → gpt-4o (still expensive!) → Success ✓
Query 1000:  "What is 2+2?" → gpt-4o-mini (learning!) → Success ✓
Query 5000:  Basic patterns established, but still refining
```

**Distribution Evolution** (illustrative):
```python
Query 1:    {"gpt-4o-mini": 0.25, "gpt-4o": 0.25, "claude": 0.50}  # Random
Query 1000: {"gpt-4o-mini": 0.40, "gpt-4o": 0.35, "claude": 0.25}  # Patterns emerging
Query 5000: {"gpt-4o-mini": 0.48, "gpt-4o": 0.32, "claude": 0.20}  # Close to optimal
```

### Phase 2: Refinement (Queries 5,001-15,000)

**Characteristics**:
- Moderate exploration (balancing exploitation with edge case discovery)
- Domain-specific patterns emerging (code vs FAQ vs creative)
- Confidence growing (distributions stabilizing)
- Approaching convergence
- Refining boundaries between simple/medium/complex

**User Experience**: Mostly good routing, improving steadily

**Example**:
```
Query 6000:  "Debug React hooks" → gpt-4o (learned: code + complex) ✓
Query 8000:  "What's your refund policy?" → gpt-4o-mini (learned: FAQ) ✓
Query 10000: "Write technical blog post" → claude-sonnet-4 (learned: creative) ✓
Query 15000: Domain patterns well-established, convergence approaching
```

**Distribution Evolution**:
```python
Query 5000:  {"gpt-4o-mini": 0.48, "gpt-4o": 0.32, "claude": 0.20}
Query 10000: {"gpt-4o-mini": 0.50, "gpt-4o": 0.30, "claude": 0.20}  # Stabilizing
Query 15000: {"gpt-4o-mini": 0.50, "gpt-4o": 0.30, "claude": 0.20}  # Near convergence
```

### Phase 3: Converged (Queries 15,001-35,000+)

**Characteristics**:
- Low exploration (95% exploitation, 5% exploration)
- Consistent routing decisions (low variance)
- High confidence in model capabilities
- Domain expertise established
- Distributions stable over time

**User Experience**: Optimal routing for this specific workload

**Example**:
```
Query 20000: "Debug this code" → gpt-4o (high confidence) ✓
Query 25000: "What is 2+2?" → gpt-4o-mini (high confidence) ✓
Query 35000: Routing fully tuned to YOUR specific workload
```

**Distribution Stability**:
```python
Query 15000: {"gpt-4o-mini": 0.50, "gpt-4o": 0.30, "claude": 0.20}
Query 25000: {"gpt-4o-mini": 0.50, "gpt-4o": 0.30, "claude": 0.20}  # Stable
Query 35000: {"gpt-4o-mini": 0.50, "gpt-4o": 0.30, "claude": 0.20}  # Converged

# Variance over last 20k queries: < 2% (proof of convergence)
```

## Training Data = Feedback Signals

### Explicit Feedback (User Ratings)

When users provide ratings:

```python
feedback = Feedback(
    quality_score=0.95,    # 0-1 scale (0.95 = 95% quality)
    user_rating=5,         # 1-5 stars
    met_expectations=True  # Boolean
)

# Convert to reward (0-1 scale)
reward = (
    quality_score * 0.6 +           # 60% weight on quality
    (user_rating / 5) * 0.4         # 40% weight on rating
) * 0.7  # 70% weight for explicit feedback

# Example: (0.95 * 0.6 + 1.0 * 0.4) * 0.7 = 0.665
```

### Implicit Feedback (Behavioral Signals)

Automatic signals from system behavior:

```python
feedback = ImplicitFeedback(
    error_occurred=False,        # No errors
    latency_seconds=0.8,         # Fast response (< 10s)
    latency_tolerance="high",    # User happy with speed
    retry_detected=False         # Didn't retry query
)

# Convert to reward with priority rules:
if error_occurred:
    reward = 0.0  # Hard failure
elif retry_detected:
    reward = 0.3  # Strong negative signal
else:
    # Latency-based reward
    reward = {
        "high": 0.9,    # < 10s
        "medium": 0.7,  # 10-30s
        "low": 0.5      # > 30s
    }[latency_tolerance]

reward *= 0.3  # 30% weight for implicit feedback

# Example: 0.9 * 0.3 = 0.27
```

### Combined Weighted Feedback

```python
# Total reward combines both signals
total_reward = explicit_reward + implicit_reward

# Example:
# Explicit: 0.665 (from rating)
# Implicit: 0.27 (from fast latency)
# Total: 0.935 → Strong success signal

# Update bandit
if total_reward >= 0.7:  # Success threshold
    model.alpha += 1  # Increment success count
else:
    model.beta += 1   # Increment failure count
```

## No Training Corpus Needed

**Key Advantage**: The bandit learns from YOUR specific workload, not generic benchmarks.

**Why this matters**:
- Different users have different query distributions
- Customer support queries ≠ code generation queries ≠ creative writing
- Generic training wouldn't work - your patterns are unique

**Example Workload Differences**:

```python
# User A: Customer Support
query_distribution = {
    "simple_faq": 60%,      # "Where is my order?"
    "moderate": 30%,        # "How do I return this?"
    "complex": 10%          # "Explain your refund policy"
}
# Optimal: Route 60% to gpt-4o-mini → 40% cost savings

# User B: Code Generation
query_distribution = {
    "simple_faq": 10%,      # "What is Python?"
    "moderate": 30%,        # "Write a function to..."
    "complex": 60%          # "Debug this architecture..."
}
# Optimal: Route 60% to gpt-4o → Different pattern!

# The bandit learns YOUR pattern, not generic expectations
```

## The Data Moat

**Competitive Advantage**: The more you use Conduit, the smarter it gets for YOUR workload.

```python
# New deployment: Generic routing (not optimized)
cost_per_1000_queries = $5.00

# After 1000 queries: Learned YOUR patterns
cost_per_1000_queries = $2.50  # 50% savings

# After 10,000 queries: Highly optimized for YOUR use case
cost_per_1000_queries = $2.25  # Even better!

# Competitor starts from scratch: Back to $5.00
# Your data creates switching costs
```

**This is the moat**: Your usage data makes Conduit increasingly valuable to you specifically.

## Convergence Metrics

### How to Measure Convergence

```python
# Track distribution stability over rolling windows
window_size = 100

# Calculate variance in model selection
recent_selections = last_100_queries.model_counts()
variance = std(recent_selections)

# Converged when variance stabilizes
if variance < threshold:
    print("Bandit has converged!")
```

### Expected Convergence Timeline

**Realistic Timeline for Production Workloads**:

- **Query 1-1,000**: Initial exploration, random to semi-random routing
- **Query 1,000-5,000**: Basic patterns emerging, high variance
- **Query 5,000-10,000**: Domain patterns visible, approaching convergence
- **Query 10,000-15,000**: Near convergence, distributions stabilizing
- **Query 15,000+**: Converged, < 2% variance over time

**With Cold Start Solutions** (informed priors + heuristics):
- Can reduce cold start period by 30-50%
- Convergence at 7,500-12,000 queries (instead of 15,000)
- Better initial routing quality

**Without Cold Start Solutions**:
- Full random exploration initially
- Convergence at 15,000-20,000 queries
- More costly learning phase

**Factors Affecting Convergence Speed**:
- **Workload diversity**: More diverse = slower convergence (4+ domains = 15k queries)
- **Feedback quality**: Better signals = faster convergence (implicit feedback helps!)
- **Model differences**: Bigger capability gaps = faster convergence (budget vs premium clear)
- **Number of models**: More models = slower convergence (4 models = 15k, 2 models = 8k)
- **Domain count**: More domains = slower convergence (need examples in each)

**Proof of Convergence**:
Measure variance in model distribution over rolling 1000-query windows:
- Converging: Variance decreasing over time
- Converged: Variance < 2% for 5,000+ consecutive queries

## Implementation Details

### Algorithm-Specific Implementations

Each algorithm has different internal mechanics. See [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) for comprehensive technical details.

**Thompson Sampling Example** (non-contextual):
```python
# conduit/engines/bandits/thompson_sampling.py
class ThompsonSamplingBandit:
    def __init__(self, arms: list[ModelArm]):
        self.arm_states = {
            arm.model_id: BetaState(alpha=1.0, beta=1.0)  # Uniform priors
            for arm in arms
        }

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        # Sample from each Beta distribution (ignores features)
        samples = {
            model_id: np.random.beta(state.alpha, state.beta)
            for model_id, state in self.arm_states.items()
        }
        return self.arms[max(samples, key=samples.get)]

    async def update(self, feedback: BanditFeedback, features: QueryFeatures):
        reward = feedback.calculate_reward()
        if reward >= 0.7:  # Success threshold
            self.arm_states[feedback.model_id].alpha += 1.0
        else:
            self.arm_states[feedback.model_id].beta += 1.0
```

**LinUCB Example** (contextual, production recommended):
```python
# conduit/engines/bandits/linucb.py
class LinUCBBandit:
    def __init__(self, arms: list[ModelArm], alpha: float = 1.0):
        self.arm_states = {
            arm.model_id: LinUCBState(
                A=np.identity(feature_dim),  # d×d matrix
                b=np.zeros((feature_dim, 1))  # d×1 vector
            )
            for arm in arms
        }

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        # Uses features for contextual decision
        x = self._extract_features(features)  # 387-dim vector

        ucb_values = {}
        for model_id, state in self.arm_states.items():
            theta = np.linalg.solve(state.A, state.b)  # Ridge regression
            ucb = theta.T @ x + self.alpha * np.sqrt(x.T @ np.linalg.inv(state.A) @ x)
            ucb_values[model_id] = ucb[0, 0]

        return self.arms[max(ucb_values, key=ucb_values.get)]

    async def update(self, feedback: BanditFeedback, features: QueryFeatures):
        x = self._extract_features(features)
        reward = feedback.calculate_reward()

        # Update matrices
        self.arm_states[feedback.model_id].A += x @ x.T
        self.arm_states[feedback.model_id].b += reward * x
```

### Key Parameters

```python
# Success threshold: When is a query "successful"?
success_threshold = 0.7  # 70% reward or higher = success

# Initial priors: Where do we start?
alpha_init = 1.0  # Uniform (no prior knowledge)
beta_init = 1.0

# Feedback weights: How to balance signals?
explicit_weight = 0.7  # User ratings: 70%
implicit_weight = 0.3  # Behavioral signals: 30%
```

## Production Considerations

### Monitoring Learning Progress

```python
# Track metrics to validate learning
metrics = {
    "queries_processed": 1000,
    "cost_per_query": 0.0025,  # $2.50 per 1000
    "avg_latency": 1.2,        # seconds
    "error_rate": 0.05,        # 5%

    # Bandit-specific metrics
    "model_distribution": {
        "gpt-4o-mini": 0.60,   # 60% of queries
        "gpt-4o": 0.30,
        "claude-sonnet-4": 0.10
    },
    "exploration_rate": 0.05,  # 5% random exploration
    "convergence_score": 0.92  # High = stable routing
}
```

### When to Reset the Bandit

Consider resetting if:
- Workload changes dramatically (customer support → code generation)
- New models added to the pool
- Pricing changes significantly
- Model capabilities change (GPT-5 released, etc.)

```python
# Soft reset: Keep some prior knowledge
bandit.reset(retain_fraction=0.5)  # Keep 50% of learned distributions

# Hard reset: Start from scratch
bandit.reset(retain_fraction=0.0)  # Back to uniform priors
```

## References

- **Algorithm Details**: [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) - Comprehensive reference for all 6 algorithms
- **Implementation Files**: `conduit/engines/bandits/*.py` - All algorithm implementations
- **Feedback System**: [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) - Automatic behavioral signals
- **Cold Start Solutions**: [COLD_START.md](COLD_START.md) - Reducing sample requirements
- **Hybrid Routing**: [HYBRID_ROUTING.md](HYBRID_ROUTING.md) - UCB1→LinUCB warm start
- **Benchmark Strategy**: [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) - Benchmarking methodology
