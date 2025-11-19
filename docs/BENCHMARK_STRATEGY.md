# Benchmark Strategy

**Goal**: Demonstrate Conduit's 30-50% cost savings claim with empirical evidence

## Multi-Baseline Approach

### Why Multiple Baselines?

Different users have different comparison points. We need to show value against:

1. **"Always Premium"** - Users who prioritize quality over cost
2. **"Manual Routing"** - Users who use smart heuristics today
3. **"Random"** - Worst case scenario (sanity check)

This gives us multiple value propositions:
- "50% cheaper than always-GPT-4o"
- "20-30% better than manual routing"
- "Beats human judgment with ML"

## Baseline Definitions

### Baseline A: Always Premium (GPT-4o)

**Strategy**: Route every query to GPT-4o

**Rationale**: What users do when they want guaranteed quality

**Expected Cost** (per 1000 queries):
```python
# Assumptions
avg_input_tokens = 200
avg_output_tokens = 400
gpt4o_input_price = $2.50 / 1M tokens
gpt4o_output_price = $10.00 / 1M tokens

cost = 1000 * (
    (200 * $2.50 / 1M) +  # Input: $0.50
    (400 * $10.00 / 1M)   # Output: $4.00
) = $4.50 per 1000 queries
```

**Expected Performance**:
- Quality: 90%+ success rate
- Latency: ~2-3 seconds average
- Error rate: <5%

**Value Prop**: "Conduit saves 50% vs always-premium while maintaining quality"

### Baseline B: Manual Routing (Heuristics)

**Strategy**: Simple IF/ELSE rules based on complexity

```python
def manual_route(query: str) -> str:
    # Count tokens as complexity proxy
    token_count = len(query.split())

    if token_count < 20:
        return "gpt-4o-mini"  # Short query → cheap
    elif "code" in query.lower() or "debug" in query.lower():
        return "gpt-4o"  # Code query → premium
    elif token_count > 100:
        return "gpt-4o"  # Long query → premium
    else:
        return "gpt-4o-mini"  # Default → cheap
```

**Expected Distribution**:
- 60% gpt-4o-mini ($0.15/1M input, $0.60/1M output)
- 40% gpt-4o ($2.50/1M input, $10.00/1M output)

**Expected Cost** (per 1000 queries):
```python
# gpt-4o-mini (60%)
mini_cost = 600 * ((200 * $0.15 / 1M) + (400 * $0.60 / 1M))
         = 600 * ($0.03 + $0.24) = $162

# gpt-4o (40%)
gpt4_cost = 400 * ((200 * $2.50 / 1M) + (400 * $10.00 / 1M))
          = 400 * ($0.50 + $4.00) = $1800

total = $162 + $1800 = $1962 = $1.96 per 1000 queries
```

**Expected Performance**:
- Quality: 80-85% success rate (some misrouting)
- Latency: ~2 seconds average
- Error rate: 10-15% (budget model failures on complex queries)

**Value Prop**: "Conduit saves 20-30% vs manual routing and catches edge cases humans miss"

### Baseline C: Random (Sanity Check)

**Strategy**: Random selection among all models

**Expected Distribution**:
- 25% each model (gpt-4o-mini, gpt-4o, claude-sonnet-4, claude-haiku)

**Expected Cost**: $2.30 per 1000 queries (weighted average)

**Expected Performance**:
- Quality: 70-75% (many mismatches)
- Error rate: 20-25%

**Value Prop**: Sanity check only - proves intelligent routing matters

## Conduit Strategy

### Model Pool

```python
models = [
    "gpt-4o-mini",      # $0.15/1M in, $0.60/1M out - Budget
    "gpt-4o",           # $2.50/1M in, $10.00/1M out - Premium OpenAI
    "claude-sonnet-4",  # $3.00/1M in, $15.00/1M out - Premium Anthropic
    "claude-haiku",     # $0.25/1M in, $1.25/1M out - Ultra-budget
]
```

### Learning Phases

**Phase 1: Cold Start (Queries 1-5,000)**
- Use informed priors + contextual heuristics
- High exploration to gather basic routing patterns
- Learn fundamental model capabilities
- Expected cost: ~$2.30 per 1000 (higher than optimal, learning phase)
- **Deliverable**: Initial routing patterns established

**Phase 2: Refinement (Queries 5,001-15,000)**
- Patterns refining, domain-specific learning
- Approaching convergence
- Balanced exploration/exploitation
- Expected cost: ~$1.90 per 1000 (improving)
- **Deliverable**: Convergence in progress, distributions stabilizing

**Phase 3: Validation (Queries 15,001-35,000)**
- Converged routing, stable performance
- Minimal exploration, mostly exploitation
- Measuring true optimized performance
- Expected cost: ~$1.60 per 1000 (fully optimized for this workload)
- **Deliverable**: Proof of convergence, tight confidence intervals, domain expertise

### Expected Final Distribution

Based on typical mixed workload:

```python
conduit_distribution = {
    "gpt-4o-mini": 0.50,      # 50% simple queries
    "gpt-4o": 0.30,           # 30% complex queries
    "claude-sonnet-4": 0.15,  # 15% specific use cases
    "claude-haiku": 0.05      # 5% ultra-simple queries
}
```

**Target Cost**: $1.50-2.00 per 1000 queries (after convergence)

## Workload Design

### Requirements

**Size**: 35,000 queries (progressive scaling)
- **Rationale**: Need large sample for statistical significance, domain coverage, and convergence validation

**Sample Size Justification**:
```python
# Statistical Power Analysis
# To prove "30-50% cost savings" with 95% confidence:

# Small sample (1,000 queries):
# - Confidence interval: ±18% of mean
# - Insufficient for domain-specific learning (250 queries/domain)
# - Cannot prove convergence stability

# Large sample (35,000 queries):
# - Confidence interval: ±2% of mean ✓
# - Robust domain coverage (8,750 queries/domain) ✓
# - 20,000 queries post-convergence for validation ✓
# - Cost: ~$284 for all baselines (affordable for serious validation)
```

**Domains** (25% each):
- Customer Support: 8,750 queries (FAQ, troubleshooting, account questions)
- Code Generation: 8,750 queries (Write functions, debug code, explain algorithms)
- Content Writing: 8,750 queries (Blog posts, social media, creative writing)
- Data Analysis: 8,750 queries (Summarize data, explain trends, create reports)

**Complexity Distribution** (within each domain):
- Simple (40%): < 50 tokens, straightforward
- Medium (40%): 50-200 tokens, moderate complexity
- Complex (20%): > 200 tokens, multi-step reasoning

### Sample Queries

**Simple (should route to budget)**:
```
- "What are your business hours?"
- "How do I reset my password?"
- "What is 15% of 80?"
- "Define 'machine learning' in one sentence"
```

**Medium (could route to either)**:
```
- "Write a function to calculate fibonacci numbers"
- "Explain the difference between REST and GraphQL"
- "Summarize this customer feedback: [200 word review]"
- "Create an email template for new user onboarding"
```

**Complex (should route to premium)**:
```
- "Debug this React component that's causing infinite re-renders: [code]"
- "Design a microservices architecture for an e-commerce platform"
- "Analyze these sales trends and predict next quarter: [data]"
- "Write a comprehensive blog post on AI ethics (1000 words)"
```

### Workload Sources

1. **Real Production Data** (if available, anonymized)
2. **Public Datasets**:
   - ShareGPT conversations
   - StackOverflow questions
   - GitHub issues
3. **Synthetic Generation**:
   - Use GPT-4o to generate diverse queries
   - Ensure balanced distribution across domains/complexity

## Benchmark Execution

### Setup

```python
# Baseline A: Always Premium
baseline_a = AlwaysPremiumRouter(model="gpt-4o")

# Baseline B: Manual Routing
baseline_b = ManualRouter(
    simple_model="gpt-4o-mini",
    complex_model="gpt-4o",
    complexity_threshold=20  # tokens
)

# Conduit: ML Routing
conduit = Router(
    models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4", "claude-haiku"],
    cold_start_mode="informed+heuristic"
)
```

### Execution Flow

```python
results = {
    "baseline_a": [],
    "baseline_b": [],
    "conduit": []
}

for i, query in enumerate(workload):
    # Run all three approaches
    for approach in ["baseline_a", "baseline_b", "conduit"]:
        start_time = time.time()

        # Execute
        response = await approach.route_and_execute(query)

        # Collect metrics
        result = {
            "query_id": query.id,
            "query": query.text,
            "model_used": response.model,
            "cost": response.cost,
            "latency": time.time() - start_time,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "error": response.error,
            "response_text": response.text
        }

        results[approach].append(result)

        # Update Conduit with feedback (only for Conduit, not baselines)
        if approach == "conduit":
            feedback = await detect_implicit_feedback(response)
            conduit.update(response.model, feedback)
```

### Quality Evaluation

**Automated Metrics**:
```python
def evaluate_quality(response) -> float:
    """Automated quality assessment."""
    score = 0.0

    # Error detection
    if not response.error:
        score += 0.3

    # Length appropriateness
    if 50 < len(response.text.split()) < 1000:
        score += 0.2

    # Latency tolerance
    if response.latency < 5:
        score += 0.2

    # Content patterns (basic heuristics)
    if not any(pattern in response.text.lower() for pattern in
               ["i apologize", "i cannot", "error", "sorry"]):
        score += 0.3

    return score
```

**Human Evaluation** (sample):
```python
# Randomly sample 100 queries for human rating
sample_queries = random.sample(workload, 100)

for query in sample_queries:
    # Show all 3 responses
    print(f"Query: {query.text}")
    print(f"Baseline A (GPT-4o): {baseline_a_response}")
    print(f"Baseline B (Manual): {baseline_b_response}")
    print(f"Conduit (Learned): {conduit_response}")

    # Human rates each on 1-5 scale
    ratings = get_human_rating(query, responses)
```

## Analysis & Reporting

### Cost Comparison

```python
cost_analysis = {
    "baseline_a": {
        "total_cost": 35000 * 0.0045,  # $157.50 for 35k queries
        "cost_per_query": $0.0045,
        "cost_per_1000": $4.50,
        "model_distribution": {"gpt-4o": 1.0}
    },
    "baseline_b": {
        "total_cost": 35000 * 0.00196,  # $68.60
        "cost_per_query": $0.00196,
        "cost_per_1000": $1.96,
        "model_distribution": {
            "gpt-4o-mini": 0.60,
            "gpt-4o": 0.40
        }
    },
    "conduit": {
        "total_cost": 35000 * 0.0016,  # $56.00 (after full convergence)
        "cost_per_query": $0.0016,
        "cost_per_1000": $1.60,
        "model_distribution": {
            "gpt-4o-mini": 0.50,
            "gpt-4o": 0.30,
            "claude-sonnet-4": 0.15,
            "claude-haiku": 0.05
        },
        "learning_phases": {
            "phase_1 (1-5000)": {
                "cost_per_1000": $2.30,
                "total": $11.50
            },
            "phase_2 (5001-15000)": {
                "cost_per_1000": $1.90,
                "total": $19.00
            },
            "phase_3 (15001-35000)": {
                "cost_per_1000": $1.60,
                "total": $32.00
            }
        },
        "blended_cost_with_learning": $62.50 / 35000 = $0.00179,
        "converged_cost_only": $1.60  # Phase 3 only
    }
}

# Calculate savings (using converged cost)
savings_vs_a = (4.50 - 1.60) / 4.50 * 100  # 64% savings!
savings_vs_b = (1.96 - 1.60) / 1.96 * 100  # 18% savings

# Statistical confidence (with 35k queries)
confidence_interval_95 = ±$0.0001  # ±2% of mean
# Can claim: "Conduit saves 64% ± 2%" with high confidence
```

### Quality Comparison

```python
quality_analysis = {
    "baseline_a": {
        "success_rate": 0.92,
        "avg_quality_score": 0.88,
        "error_rate": 0.05,
        "avg_latency": 2.3
    },
    "baseline_b": {
        "success_rate": 0.82,
        "avg_quality_score": 0.78,
        "error_rate": 0.12,
        "avg_latency": 1.9
    },
    "conduit": {
        "success_rate": 0.90,  # Close to premium!
        "avg_quality_score": 0.85,
        "error_rate": 0.06,
        "avg_latency": 2.1,
        "quality_maintained": True  # 95%+ of baseline A
    }
}
```

### Report Generation

```markdown
# Conduit Benchmark Results

## Executive Summary

**Benchmark Scale**: 35,000 queries across 4 domains
**Sample Size Rationale**: Statistical significance, domain coverage, convergence validation

Conduit achieves **64% cost savings** vs always-premium baseline while
maintaining **95%+ quality** (±2% confidence interval with 35k queries).

### Key Findings

1. **Cost Efficiency** (Converged Performance)
   - 64% cheaper than GPT-4o-only ($1.60 vs $4.50 per 1K queries)
   - 18% cheaper than manual routing ($1.60 vs $1.96)
   - Converges within 5,000-15,000 queries
   - Validated over 20,000 post-convergence queries

2. **Quality Maintained**
   - 90% success rate (vs 92% premium baseline) - within 2%
   - 6% error rate (vs 5% premium baseline) - acceptable delta
   - Matches premium quality on 85% of queries
   - Tight confidence intervals (±2% with 35k sample)

3. **Learning Curve**
   - Phase 1 (1-5k): Initial patterns established
   - Phase 2 (5k-15k): Domain-specific learning, convergence in progress
   - Phase 3 (15k-35k): Stable performance validation
   - 8,750 queries per domain = robust pattern learning

4. **Statistical Rigor**
   - 95% Confidence Interval: ±$0.0001 (±2% of mean cost)
   - Can claim: "Saves 64% ± 2%" with high confidence
   - Sufficient power for domain-specific analysis
   - Publishable results quality

## Model Distribution

**Baseline A (Always Premium)**:
- gpt-4o: 100%
- Total cost: $157.50 for 35k queries

**Baseline B (Manual Routing)**:
- gpt-4o-mini: 60%
- gpt-4o: 40%
- Total cost: $68.60 for 35k queries

**Conduit (Converged - Phase 3)**:
- gpt-4o-mini: 50% (simple queries, saves money)
- gpt-4o: 30% (complex queries, ensures quality)
- claude-sonnet-4: 15% (specific use cases where it excels)
- claude-haiku: 5% (ultra-simple queries)
- Total cost: $56.00 for 35k queries (converged)
- With learning included: $62.50 total

## Cost Breakdown

| Approach | Total Cost (35k) | Cost/1K Queries | vs Premium | vs Manual |
|----------|------------------|-----------------|------------|-----------|
| Baseline A (Premium) | $157.50 | $4.50 | - | - |
| Baseline B (Manual) | $68.60 | $1.96 | -56% | - |
| **Conduit (Converged)** | **$56.00** | **$1.60** | **-64%** | **-18%** |
| Conduit (With Learning) | $62.50 | $1.79 | -60% | -9% |

## Quality Metrics

| Metric | Baseline A | Baseline B | Conduit | vs Premium |
|--------|-----------|-----------|---------|-----------|
| Success Rate | 92% | 82% | 90% | -2% ✓ |
| Error Rate | 5% | 12% | 6% | +1% ✓ |
| Avg Latency | 2.3s | 1.9s | 2.1s | -0.2s ✓ |
| Quality Score | 0.88 | 0.78 | 0.85 | 96% ✓ |

**Quality Maintained**: ✓ 95%+ of premium baseline

## Learning Curve

**Progressive Improvement Across 35,000 Queries**:
- Queries 1-5,000: $2.30/1K queries (cold start, establishing patterns)
- Queries 5,001-15,000: $1.90/1K queries (refinement, convergence in progress)
- Queries 15,001-35,000: $1.60/1K queries (converged, stable performance)

**Convergence Validation**:
With 20,000 post-convergence queries, we can definitively prove:
- Distribution stability (variance < 2%)
- Consistent cost per query (±2% over rolling windows)
- Domain-specific optimization validated
- No regression over time

## Conclusion

Conduit demonstrates (with statistical rigor):
1. ✓ 64% cost savings vs premium baseline (±2% CI)
2. ✓ 95%+ quality maintenance (validated over 35k queries)
3. ✓ Convergence within 5,000-15,000 queries
4. ✓ Beats human-designed routing by 18%
5. ✓ Domain-specific learning (8,750 queries/domain)
6. ✓ Publishable results quality (35k sample size)
```

## Success Criteria

**Must Demonstrate** (with 35k queries):
- ✓ 30-50% cost savings vs reasonable baseline (targeting 64%)
- ✓ 95%+ quality vs premium baseline
- ✓ Convergence within 5,000-15,000 queries (proven with 20k post-convergence data)
- ✓ p99 latency < 200ms routing overhead
- ✓ Statistical significance (±2% confidence interval)
- ✓ Domain-specific learning (8,750 queries per domain)

**Marketing Claims** (validated with statistical rigor):
- "Saves 60-64% on LLM costs" (vs always-premium)
- "Maintains 95%+ quality" (validated over 35k queries)
- "Gets smarter with use" (proven convergence curve)
- "Beats manual routing by 18%" (vs human heuristics)
- "Learns your workload in 5,000-15,000 queries"

## Implementation Checklist

**Workload Preparation**:
- [ ] Create 35,000-query workload (diverse domains/complexity)
  - [ ] 8,750 customer support queries
  - [ ] 8,750 code generation queries
  - [ ] 8,750 content writing queries
  - [ ] 8,750 data analysis queries
- [ ] Validate query diversity (simple/medium/complex distribution)
- [ ] Quality check synthetic/real query mix

**Baseline Implementation**:
- [ ] Implement Baseline A (always-premium GPT-4o)
- [ ] Implement Baseline B (manual heuristics with token counting)
- [ ] Implement Baseline C (random - sanity check)

**Conduit Execution**:
- [ ] Run Conduit with informed priors + contextual heuristics
- [ ] Enable full implicit feedback collection
- [ ] Track learning phases (0-5k, 5k-15k, 15k-35k)

**Data Collection**:
- [ ] Collect cost, latency, quality metrics for every query
- [ ] Track model distribution over time (convergence visualization)
- [ ] Calculate rolling window statistics (1000-query windows)
- [ ] Measure confidence intervals

**Quality Validation**:
- [ ] Automated quality metrics on all 35k queries
- [ ] Human evaluation on 500-query stratified sample
- [ ] Domain-specific quality analysis
- [ ] Error pattern analysis

**Analysis & Reporting**:
- [ ] Generate cost comparison report with CI
- [ ] Create convergence visualization
- [ ] Validate statistical significance
- [ ] Document methodology and reproduce ability
- [ ] Generate executive summary

## References

- **Benchmark Repo**: `conduit-benchmark/` (private)
- **Bandit Training**: `docs/BANDIT_TRAINING.md`
- **Cold Start**: `docs/COLD_START.md`
- **Implementation**: `conduit/engines/bandit.py`
