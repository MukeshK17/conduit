# Conduit

ML-powered LLM routing system that learns optimal model selection for cost, latency, and quality optimization.

## Overview

Conduit uses contextual bandits (Thompson Sampling) to intelligently route queries to the optimal LLM model based on learned patterns from usage data. Unlike static rule-based routers, Conduit continuously improves routing decisions through feedback loops.

## Key Features

- **ML-Driven Routing**: Learns from usage patterns vs static IF/ELSE rules
- **Multi-Objective Optimization**: Balance cost, latency, and quality constraints
- **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Groq via PydanticAI
- **Feedback Loop**: Improves from user ratings and quality metrics
- **Cost Prediction**: Estimate costs before execution
- **A/B Testing**: Built-in experimentation framework

## Quick Start

```python
from conduit import Router
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    answer: str
    confidence: float

router = Router(
    models=["gpt-4o-mini", "gpt-4o", "claude-opus-4"],
    optimize_for="cost_quality_balanced"
)

# Automatic intelligent routing
response = await router.complete(
    prompt="What's 2+2?",
    result_type=AnalysisResult
)

# Provide feedback to improve routing
await router.feedback(
    response_id=response.id,
    quality_score=0.95
)
```

## Installation & Setup

### Prerequisites

- Python 3.10+ (3.13 recommended)
- Supabase account (free tier works)
- Redis instance (optional for Phase 2+ caching)
- LLM API keys (OpenAI, Anthropic, Google, or Groq)

### Step 1: Clone and Install

```bash
git clone https://github.com/MisfitIdeas/conduit.git
cd conduit

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Install development tools
pip install mypy black ruff pytest pytest-asyncio pytest-cov psycopg2-binary
```

**Note:** Some ML dependencies (scipy, scikit-learn) require Fortran compilers. If installation fails, you can still use the core functionality.

### Step 2: Environment Configuration

Create `.env` file:
```bash
# LLM Provider API Keys (at least one required)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here

# Supabase (required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
DATABASE_URL=postgresql://postgres.your-project:[password]@aws-0-region.pooler.supabase.com:6543/postgres

# Redis (optional - Phase 2+)
REDIS_URL=redis://localhost:6379

# Application
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Step 3: Database Setup

**Option A: Using Alembic (recommended)**
```bash
# Ensure DATABASE_URL in .env points to Supabase pooler connection
./migrate.sh
```

**Option B: Manual SQL (if pooler not available)**
```bash
# Copy SQL content and paste into Supabase SQL Editor
cat migrations/001_initial_schema.sql
```

See `migrations/DEPLOYMENT.md` for detailed migration instructions.

## Tech Stack

- **Python 3.10+**
- **PydanticAI 1.14+** (unified LLM interface)
- **FastAPI** (REST API)
- **PostgreSQL** (routing history)
- **scikit-learn** (ML algorithms)
- **sentence-transformers** (query embeddings)

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov=conduit

# Type checking
mypy conduit/

# Linting
ruff check conduit/

# Formatting
black conduit/
```

## Architecture

```
Query → Embedding → ML Routing Engine → LLM Provider → Response
   ↓                                                        ↓
   └─────────────────── Feedback Loop ─────────────────────┘
```

**Routing Process**:
1. Analyze query (embedding, features)
2. ML model predicts optimal route
3. Execute via PydanticAI
4. Collect feedback
5. Update routing model

## Documentation

See `docs/` for detailed design specifications and implementation guides.

## License

MIT License - see LICENSE file for details

## Status

**Phase**: Initial development
**Version**: 0.0.1-alpha
