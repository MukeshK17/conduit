# Industry Priors for Cold Start Optimization

This document explains the industry-wide prior knowledge system used for cold start optimization in Conduit's contextual bandit routing.

## Overview

When Conduit starts routing queries to a new model or for a new user, it faces a "cold start" problem: without historical data, the bandit algorithm has no basis for estimating model quality. Industry priors provide informed starting points based on public benchmark data.

## Data Sources

Priors are derived from two authoritative benchmark sources:

### 1. Artificial Analysis (https://artificialanalysis.ai/leaderboards/providers)

Provides comprehensive model benchmarks including:
- **Intelligence Index**: Composite score across reasoning tasks
- **MMLU Pro**: Massive Multitask Language Understanding (professional)
- **GPQA**: Graduate-level question answering
- **LiveCodeBench**: Real-time coding benchmark
- **Math Index**: Mathematical reasoning capability

### 2. Vellum LLM Leaderboard (https://www.vellum.ai/llm-leaderboard)

Provides task-specific benchmarks including:
- **GPQA Diamond**: Graduate-level science questions
- **Agentic Coding (SWE-Bench)**: Real-world software engineering tasks
- **MATH 500**: Mathematical problem solving
- **HumanEval+**: Code generation benchmark
- **IFEval**: Instruction following evaluation

## How Priors Work

Priors are expressed as Beta distribution parameters `(alpha, beta)`:

```
Quality Estimate = alpha / (alpha + beta)
Prior Strength = alpha + beta
```

For example, `(8500, 1500)` means:
- Quality estimate: 8500 / 10000 = 85%
- Prior strength: 10000 (high confidence)

Higher prior strength means the algorithm trusts the prior more and explores less initially.

## Context-Specific Priors

Different query contexts have different optimal models:

### Code Context
Best for: Code generation, debugging, technical queries

Models excel differently at coding tasks. Based on LiveCodeBench and SWE-Bench:
- Claude Sonnet 4.5 leads on SWE-Bench (agentic coding)
- GPT-5.1 excels at code completion and generation

### Creative Context
Best for: Creative writing, storytelling, content generation

Based on creative writing benchmarks and qualitative assessments:
- Claude models generally preferred for nuanced creative work
- Style and voice consistency varies by model

### Analysis Context
Best for: Analytical reasoning, comparison, evaluation

Based on GPQA and reasoning benchmarks:
- Premium models (GPT-5.1, Claude Opus 4.5) excel at complex analysis
- Significant quality gap vs smaller models on reasoning tasks

### Simple QA Context
Best for: Factual questions, straightforward queries

Based on MMLU and general knowledge benchmarks:
- Smaller models often sufficient for simple factual queries
- Cost-effectiveness favors mini/haiku models

### General Context
Default priors for unclassified queries, balanced across capabilities.

## Updating Priors

Priors should be updated periodically as:
1. New benchmark data becomes available
2. Models are updated or deprecated
3. New models are released

### Manual Update

Edit `conduit.yaml` directly:

```yaml
priors:
  code:
    claude-sonnet-4.5: 0.92  # 92% quality estimate
    gpt-5.1: 0.88            # 88% quality estimate
```

### Sync Script

Run the sync script to fetch latest benchmark data:

```bash
python scripts/sync_priors.py --output conduit.yaml
```

The script:
1. Fetches current benchmark data from Artificial Analysis and Vellum
2. Maps benchmark scores to Beta distribution parameters
3. Applies context-specific weighting
4. Outputs updated prior configuration

## Methodology: Benchmark to Prior Conversion

### Step 1: Normalize Benchmark Scores

Each benchmark score is normalized to 0-1 range:
```
normalized = (score - min_score) / (max_score - min_score)
```

### Step 2: Weight by Context Relevance

Different benchmarks are weighted by context:

| Context | LiveCodeBench | GPQA | MMLU | Creative |
|---------|--------------|------|------|----------|
| code    | 0.5          | 0.2  | 0.2  | 0.1      |
| creative| 0.1          | 0.2  | 0.2  | 0.5      |
| analysis| 0.2          | 0.4  | 0.3  | 0.1      |
| simple_qa| 0.1         | 0.2  | 0.6  | 0.1      |
| general | 0.25         | 0.25 | 0.25 | 0.25     |

### Step 3: Convert to Beta Parameters

```python
def to_beta_params(quality: float, strength: int = 10000) -> tuple[int, int]:
    """Convert quality score to Beta distribution parameters.

    Args:
        quality: Quality score in range [0, 1]
        strength: Prior strength (higher = more confident)

    Returns:
        (alpha, beta) tuple for Beta distribution
    """
    alpha = int(quality * strength)
    beta = strength - alpha
    return (alpha, beta)
```

### Step 4: Validate and Clip

- Quality estimates clipped to [0.5, 0.95] range
- Prevents extreme priors that could harm exploration
- Prior strength standardized to 10000 for comparability

## Configuration Reference

### conduit.yaml Structure

```yaml
priors:
  # Context name (code, creative, analysis, simple_qa, general)
  code:
    # model_id: quality_score (0.0-1.0)
    claude-sonnet-4.5: 0.92    # Best for code (SWE Bench leader)
    claude-opus-4.5: 0.91      # Premium code generation
    gpt-5.1: 0.88              # Strong coding performance
    gpt-5: 0.85                # Good balance
    o4-mini: 0.78              # Fast and decent
    gemini-2.5-pro: 0.82       # Strong alternative
    gemini-2.0-flash: 0.72     # Quick responses
```

### Environment Variables

```bash
# Custom priors file location
CONDUIT_PRIORS_PATH=/path/to/custom_priors.yaml

# Disable priors (use uniform priors)
CONDUIT_USE_PRIORS=false

# Prior strength multiplier (default 1.0)
CONDUIT_PRIOR_STRENGTH=0.5  # Half strength = more exploration
```

## API Usage

### Load Priors Programmatically

```python
from conduit.core import load_context_priors

# Load priors for code context
code_priors = load_context_priors("code")
# Returns: {"claude-sonnet-4.5": (9200, 800), "gpt-5.1": (8800, 1200), ...}

# Use in Thompson Sampling
for model_id, (alpha, beta) in code_priors.items():
    bandit.set_prior(model_id, alpha, beta)
```

### Dynamic Prior Loading

```python
from conduit.core.priors import PriorLoader

# Load from custom source
loader = PriorLoader()
priors = await loader.fetch_from_benchmarks()

# Apply to bandit
bandit.update_priors(priors)
```

## Benchmark Data Snapshot

Last updated: 2025-11

### Current Model Rankings (by context)

| Context | Top Models |
|---------|------------|
| Code | claude-sonnet-4.5 (92%), claude-opus-4.5 (91%), gpt-5.1 (88%) |
| Creative | claude-opus-4.5 (94%), claude-sonnet-4.5 (90%), gpt-5.1 (86%) |
| Analysis | claude-opus-4.5 (92%), gpt-5.1 (89%), claude-sonnet-4.5 (88%) |
| Simple QA | o4-mini (90%), gemini-2.0-flash (88%), gpt-5 (85%) |
| General | gpt-5.1 (88%), claude-opus-4.5 (87%), claude-sonnet-4.5 (85%) |

*Note: Quality scores from conduit.yaml priors configuration.*

## Limitations

1. **Benchmark-Task Mismatch**: Benchmarks may not perfectly predict performance on your specific queries
2. **Version Drift**: Model versions change; benchmarks may be outdated
3. **Context Classification**: Prior selection depends on accurate context detection
4. **Provider Differences**: Same model via different providers may vary

## Best Practices

1. **Start with industry priors** for new deployments
2. **Monitor actual performance** and compare to prior predictions
3. **Reduce prior strength** if your use case differs from benchmarks
4. **Update regularly** as new benchmark data becomes available
5. **Consider custom priors** for specialized domains
