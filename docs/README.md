# Conduit Router Documentation

Comprehensive technical documentation for understanding and working with the Conduit Router.

---

## Quick Start Paths

### 🚀 New to the Conduit Router?

**Learn the Basics** (15 minutes):
1. [BANDIT_TRAINING.md](BANDIT_TRAINING.md) - Understand why contextual bandits are the right abstraction
2. [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) - See what models are available (71+ models, auto-detected)
3. `../examples/01_quickstart/hello_world.py` - Run your first query (5 lines of code)

**Key Insight**: Zero-shot deployment. No pre-training required!

---

### 🎯 Want to Optimize Performance?

**Reduce Sample Requirements** (30 minutes):
1. [COLD_START.md](COLD_START.md) - Understand the cold start problem
2. [HYBRID_ROUTING.md](HYBRID_ROUTING.md) - Implement UCB1→LinUCB warm start (30% faster)
3. [PCA_GUIDE.md](PCA_GUIDE.md) - Add dimensionality reduction (75% sample reduction)
4. `../examples/03_optimization/` - Working examples

**Combined Impact**: 1,500-2,500 queries to production (vs 10,000+ baseline)

---

### 🔌 Want to Integrate?

**LiteLLM Plugin** (20 minutes):
1. [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) - Understand the plugin strategy
2. `../conduit_litellm/` - LiteLLM integration module
3. `../docker-compose.openwebui.yml` - Full stack with Open WebUI

**Result**: Access 100+ LLM providers with ML-powered routing

---

### 📊 Want to Measure Performance?

**Benchmarking Guide**:
1. [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) - Methodology for measuring routing performance
2. Create workload matching your use case
3. Run comparison: Always Premium vs Manual vs Conduit Router
4. Measure cost, quality, and latency metrics

**What to Measure**: Cost per query, quality scores, latency, and model selection patterns

---

## Core Concepts

### [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) - Dynamic Pricing & Model Discovery

**Key Topics**:
- Auto-fetch 71+ models from llm-prices.com (24h cache)
- `supported_models()` - see what the Conduit Router can use
- `available_models()` - see what YOU can use (auto-detects API keys)
- Zero maintenance (no hard-coded pricing tables)
- Graceful degradation with fallback pricing

**Read this first if**: You want to understand model availability and pricing

---

### [BANDIT_TRAINING.md](BANDIT_TRAINING.md) - How Learning Works

**Key Topics**:
- Why contextual bandits (not full RL) - the right abstraction for LLM routing
- Online learning from feedback (no pre-training needed)
- Learning phases: Cold Start → Learning → Converged
- Feedback signals: Explicit (ratings) + Implicit (behavior)
- The "Data Moat" competitive advantage

**Read this if**: You want to understand how the Conduit Router learns from usage

---

### [COLD_START.md](COLD_START.md) - The Cold Start Problem

**Key Topics**:
- Problem definition: Making good decisions before learning
- 7 solution approaches with pros/cons
- Implemented solutions: Hybrid Routing + PCA (85-90% sample reduction)
- Recommended strategy: Informed Priors + Contextual Heuristics
- Expected convergence in 1,500-2,500 queries (vs 10,000+ without)

**Read this if**: You want to minimize poor routing during initial queries

---

### [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) - Benchmarking Methodology

**Key Topics**:
- Multi-baseline approach (Always Premium, Manual Routing, Random)
- Workload design for measuring routing performance
- Quality validation methodology
- Report generation

**Read this if**: You want to measure routing performance and compare strategies

---

## System Architecture

### [ARCHITECTURE.md](ARCHITECTURE.md) - System Design

High-level system architecture and component interactions.

**Covers**:
- Layer architecture (API → Router → Executor → Providers)
- Database schema (PostgreSQL)
- Caching strategy (Redis)
- Feedback collection and integration

---

### [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) - Observability Trinity

Detailed documentation of the implicit feedback system.

**Three Detection Systems**:
1. **Error Detection** - Model failures, empty responses, error patterns
2. **Latency Tracking** - User patience tolerance categorization
3. **Retry Detection** - Semantic similarity-based duplicate detection

**Weighting**: 70% explicit feedback + 30% implicit signals

---

### [MEASUREMENT_STRATEGY.md](MEASUREMENT_STRATEGY.md) - Quality Assessment & Tracking

Three-tier measurement strategy for routing quality and performance tracking.

**Tier 1 - Core Metrics** ✅:
- Regret calculation (vs oracle, random, always-best)
- Quality trends (7-day moving average)
- Cost efficiency metrics

**Tier 2 - Automated Evaluation** (Partial ✅):
- Arbiter LLM-as-judge integration ✅ (commits: 0598f61, a1ceb96)
- pgvector embeddings database 📋

**Tier 3 - Advanced Analysis** 📋:
- Loom batch pipelines for A/B testing
- Query pattern clustering
- Real-time metrics dashboard

**Related**: See [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) for automatic feedback loop from LiteLLM

---

### [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) - Algorithm Reference

Comprehensive documentation of all bandit algorithms.

**Algorithms**:
- **Contextual Thompson Sampling** - Bayesian linear regression (best for diverse queries)
- **LinUCB** - Ridge regression with UCB (best for context-aware routing)
- **Thompson Sampling** - Beta distributions (good exploration/exploitation)
- **UCB1** - Upper confidence bounds (fast, non-contextual)
- **Epsilon-Greedy** - Simple exploration (baseline)
- **Hybrid Routing** - UCB1→LinUCB warm start (30% faster convergence)

**Related**: [COLD_START.md](COLD_START.md) for sample efficiency strategies

---

### [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) - LiteLLM Integration Strategy

Strategic analysis and implementation plans for LiteLLM integration.

**Paths**:
- **Path 1** (Recommended): Conduit Router as LiteLLM routing strategy plugin
- **Path 2**: LiteLLM as Conduit Router execution backend

**Benefits**:
- Access to 100+ providers through LiteLLM ecosystem
- OpenAI-compatible API
- Docker Compose setup with Open WebUI

---

## Sample Efficiency Guides

### [HYBRID_ROUTING.md](HYBRID_ROUTING.md) - UCB1→LinUCB Warm Start

**Performance**: 30% faster convergence than pure LinUCB

**Strategy**:
1. Phase 1 (0-2,000 queries): UCB1 (non-contextual, fast exploration)
2. Phase 2 (2,000+ queries): LinUCB (contextual, warm-started from UCB1)

**Sample Requirements**: 2,000-3,000 queries vs 10,000+ for pure LinUCB

---

### [PCA_GUIDE.md](PCA_GUIDE.md) - Dimensionality Reduction

**Performance**: 75% sample reduction for LinUCB convergence

**Compression**: 387 → 67 dimensions (82% reduction)
- 384 embedding dims → 64 PCA components (95%+ variance)
- 3 metadata dims → preserved (tokens, complexity, confidence)

**Sample Requirements**: 17,000 queries vs 68,000 for full features

---

### Combined Impact

| Approach | Sample Requirement | Improvement |
|----------|-------------------|-------------|
| Pure LinUCB (387d) | 10,000-15,000 queries | Baseline |
| Hybrid (387d) | 2,000-3,000 queries | 70-85% reduction |
| PCA (67d) | 3,000-5,000 queries | 60-70% reduction |
| **Hybrid + PCA (67d)** | **1,500-2,500 queries** | **85-90% reduction** |

---

## Quick Navigation

### I want to understand...

**...what models I can use**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md)

**...how pricing stays current**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) (Dynamic Pricing section)

**...how the Conduit Router learns**: → [BANDIT_TRAINING.md](BANDIT_TRAINING.md)

**...why it doesn't need pre-training**: → [BANDIT_TRAINING.md](BANDIT_TRAINING.md) (Why Contextual Bandits section)

**...how to reduce cold start problems**: → [COLD_START.md](COLD_START.md)

**...sample efficiency (PCA, Hybrid Routing)**: → [HYBRID_ROUTING.md](HYBRID_ROUTING.md) + [PCA_GUIDE.md](PCA_GUIDE.md)

**...how to benchmark routing performance**: → [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md)

**...how implicit feedback works**: → [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md)

**...the bandit algorithms**: → [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md)

**...LiteLLM integration options**: → [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md)

**...measurement and quality tracking**: → [MEASUREMENT_STRATEGY.md](MEASUREMENT_STRATEGY.md)

**...the overall system design**: → [ARCHITECTURE.md](ARCHITECTURE.md)

---

### I want to implement...

**...model discovery**: → [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) (API Reference) + `examples/01_quickstart/hello_world.py`

**...hybrid routing**: → [HYBRID_ROUTING.md](HYBRID_ROUTING.md) + `Router(use_hybrid=True)`

**...PCA reduction**: → [PCA_GUIDE.md](PCA_GUIDE.md) + `examples/03_optimization/`

**...LiteLLM integration**: → [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) + `conduit_litellm/`

**...a benchmark comparison**: → [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) (Execution Flow)

**...implicit feedback collection**: → [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) (Usage Examples)

---

## Implementation Paths

### Path 1: Basic Routing (15 minutes)
```
Read: BANDIT_TRAINING.md → MODEL_DISCOVERY.md
Code: examples/01_quickstart/hello_world.py
Test: Run your first query
```

### Path 2: Optimized Routing (1 hour)
```
Read: COLD_START.md → HYBRID_ROUTING.md → PCA_GUIDE.md
Code: examples/03_optimization/ (working examples)
Code: Router(use_hybrid=True, use_pca=True)
Test: Verify 85-90% sample reduction
```

### Path 3: LiteLLM Integration (30 minutes)
```
Read: LITELLM_INTEGRATION.md
Code: conduit_litellm/ (integration module)
Deploy: docker-compose.openwebui.yml (full stack)
Test: Access via OpenAI-compatible API
```

### Path 4: Production Deployment (2 hours)
```
Read: ARCHITECTURE.md → IMPLICIT_FEEDBACK.md
Setup: PostgreSQL + Redis
Configure: Environment variables (.env)
Deploy: Docker or direct Python
Monitor: Quality, cost, latency metrics
```

---

## Key Insights

### Contextual Bandits vs Full RL

| Aspect | Contextual Bandits | Full RL (MDP) |
|--------|-------------------|---------------|
| State transitions | None - independent decisions | Actions affect future states |
| Horizon | Single step | Multi-step episodes |
| Credit assignment | Immediate reward | Delayed rewards |
| Complexity | O(actions × features) | O(states × actions × transitions) |

**Bottom Line**: LLM routing has no state transitions. Bandits are the right abstraction.

---

### Cold Start Solutions (Recommended)

**Implemented** (Production-Ready):
1. **Hybrid Routing** - UCB1→LinUCB warm start (30% faster)
2. **PCA Reduction** - 75% sample reduction

**Combined**: 1,500-2,500 queries to production (vs 10,000+ baseline)

**Future** (See COLD_START.md):
- Informed Priors (industry/domain knowledge)
- Contextual Heuristics (query feature-based routing)
- Transfer Learning (share learnings across customers)

---

## Documentation Status

| Document | Purpose | Status |
|----------|---------|--------|
| **Core Concepts** | | |
| [BANDIT_TRAINING.md](BANDIT_TRAINING.md) | How learning works, why bandits | ✅ Complete |
| [BANDIT_ALGORITHMS.md](BANDIT_ALGORITHMS.md) | All 9 algorithm implementations | ✅ Complete |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and components | ✅ Complete |
| **Optimization** | | |
| [COLD_START.md](COLD_START.md) | Reducing early routing mistakes | ✅ Complete |
| [HYBRID_ROUTING.md](HYBRID_ROUTING.md) | UCB1→LinUCB warm start | ✅ Complete |
| [PCA_GUIDE.md](PCA_GUIDE.md) | Dimensionality reduction | ✅ Complete |
| [PRIORS.md](PRIORS.md) | Industry priors for cold start | ✅ Complete |
| **Integration** | | |
| [LITELLM_INTEGRATION.md](LITELLM_INTEGRATION.md) | LiteLLM plugin strategy | ✅ Complete |
| [MODEL_DISCOVERY.md](MODEL_DISCOVERY.md) | Dynamic pricing & model detection | ✅ Complete |
| [EMBEDDING_PROVIDERS.md](EMBEDDING_PROVIDERS.md) | Embedding provider options | ✅ Complete |
| **Operations** | | |
| [BOOTSTRAP_STRATEGY.md](BOOTSTRAP_STRATEGY.md) | Configuration and startup | ✅ Complete |
| [IMPLICIT_FEEDBACK.md](IMPLICIT_FEEDBACK.md) | Automatic feedback detection | ✅ Complete |
| [MEASUREMENT_STRATEGY.md](MEASUREMENT_STRATEGY.md) | Quality tracking and metrics | ✅ Complete |
| [BENCHMARK_STRATEGY.md](BENCHMARK_STRATEGY.md) | Benchmarking methodology | ✅ Complete |
| **Support** | | |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Debugging common issues | ✅ Complete |
| [FAQ.md](FAQ.md) | Frequently asked questions | ✅ Complete |
| **Quick Reference** | | |
| [CONFIGURATION.md](CONFIGURATION.md) | All config options in one place | ✅ Complete |
| [RECIPES.md](RECIPES.md) | Copy-paste code for common tasks | ✅ Complete |

---

## Glossary

| Term | Definition |
|------|------------|
| **Arm** | A model option in the bandit algorithm (e.g., gpt-4o-mini is one "arm") |
| **Bandit** | Algorithm that balances trying new options (exploration) vs using known-good options (exploitation) |
| **Contextual Bandit** | Bandit that uses query features (context) to make decisions, not just historical rewards |
| **Cold Start** | The period before the bandit has enough data to make confident routing decisions |
| **Convergence** | When the bandit has learned enough to make stable, near-optimal routing decisions |
| **Exploration** | Trying different models to learn their performance |
| **Exploitation** | Using the model known to work best based on past data |
| **Features** | Query characteristics used for routing: embedding (384 dims), token count, complexity, domain |
| **Hybrid Routing** | Two-phase strategy: UCB1 (fast warmup) then LinUCB (contextual refinement) |
| **LinUCB** | Linear Upper Confidence Bound - contextual bandit using ridge regression |
| **PCA** | Principal Component Analysis - reduces 387 features to 67 for faster learning |
| **Priors** | Initial beliefs about model quality, used to speed up cold start |
| **Regret** | Difference between optimal choice and actual choice; lower is better |
| **Reward** | Composite score (quality + cost + latency) used to train the bandit |
| **Thompson Sampling** | Bayesian bandit algorithm that samples from probability distributions |
| **UCB1** | Upper Confidence Bound - non-contextual bandit with logarithmic exploration |

---

## Contributing

When updating documentation:

1. **Keep it practical**: Focus on implementation, not just theory
2. **Use code examples**: Show don't tell
3. **Update README.md**: Add your new doc to the navigation
4. **Cross-reference**: Add "See Also" sections linking related docs
5. **Date your updates**: Keep "Last Updated" current

---

## Questions?

- **Technical Implementation**: See `../CLAUDE.md` in project root
- **Development Workflow**: See `../CLAUDE.md` in project root
- **Strategic Decisions**: See `../notes/` directory
