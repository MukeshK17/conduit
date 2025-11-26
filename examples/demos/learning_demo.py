"""Learning demo showing Conduit Router adapting over time.

This script demonstrates how Conduit learns to route queries to optimal models
by showing routing decisions changing as the router gains experience.

Run with:
    python examples/demos/learning_demo.py

For GIF recording:
    asciinema rec learning_demo.cast
    # Or use: termtosvg learning_demo.svg
"""

import asyncio
import random
import time
from typing import List

from rich.console import Console
from rich.live import Live
from rich.table import Table

from conduit.core.models import Query
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router


class LearningDemo:
    """Interactive demo showing Conduit learning over time."""

    def __init__(self):
        """Initialize demo."""
        self.router = Router()
        self.executor = ModelExecutor()
        self.console = Console()
        self.history: List[dict] = []

    def generate_queries(self, count: int = 50) -> List[dict]:
        """Generate diverse queries for demonstration.

        Args:
            count: Number of queries to generate

        Returns:
            List of query dictionaries with text and complexity
        """
        simple_queries = [
            "What is 2+2?",
            "What is the capital of France?",
            "What is Python?",
            "Hello, how are you?",
            "What is the weather?",
            "What time is it?",
            "What is 5*5?",
            "What is the meaning of life?",
            "What is a variable?",
            "What is a function?",
        ]

        complex_queries = [
            "Explain quantum computing and its applications in cryptography",
            "Write a comprehensive guide to machine learning model evaluation",
            "Design a distributed system architecture for handling 1M requests/second",
            "Explain the mathematical foundations of neural networks",
            "Compare and contrast different database architectures",
            "Write a detailed analysis of the CAP theorem",
            "Explain how transformers work in natural language processing",
            "Design a microservices architecture for an e-commerce platform",
            "Explain the principles of functional programming",
            "Write a guide to optimizing database queries",
        ]

        queries = []
        for i in range(count):
            # Mix simple and complex queries
            if i < count * 0.6:  # 60% simple queries
                text = random.choice(simple_queries)
                complexity = "simple"
            else:  # 40% complex queries
                text = random.choice(complex_queries)
                complexity = "complex"

            queries.append({"text": text, "complexity": complexity})

        return queries

    async def simulate_query(self, query_text: str, complexity: str) -> dict:
        """Simulate a query execution.

        Args:
            query_text: Query text
            complexity: Query complexity ("simple" or "complex")

        Returns:
            Dictionary with query results
        """
        query = Query(text=query_text)

        # Route query
        decision = await self.router.route(query)

        # Simulate execution (in real usage, this would call the model)
        # For demo purposes, we'll simulate cost and quality based on model
        model_id = decision.selected_model

        # Simulate cost (cheaper models for simple queries)
        if "mini" in model_id.lower() or "haiku" in model_id.lower():
            cost = 0.0001 if complexity == "simple" else 0.0005
            quality = 0.85 if complexity == "simple" else 0.70
        else:
            cost = 0.001 if complexity == "simple" else 0.002
            quality = 0.95 if complexity == "simple" else 0.90

        # Simulate latency
        latency = 0.5 if complexity == "simple" else 1.5

        # Update router with feedback (simulated)
        from conduit.engines.bandits.base import BanditFeedback

        feedback = BanditFeedback(
            model_id=model_id,
            cost=cost,
            quality_score=quality,
            latency=latency,
        )

        # Get features for update
        features = await self.router.analyzer.analyze(query)
        await self.router.hybrid_router.update(feedback, features)

        return {
            "query": query_text[:50] + "..." if len(query_text) > 50 else query_text,
            "model": model_id,
            "cost": cost,
            "quality": quality,
            "latency": latency,
            "complexity": complexity,
        }

    def create_table(self, current_query_num: int, total_queries: int) -> Table:
        """Create Rich table showing current state.

        Args:
            current_query_num: Current query number
            total_queries: Total number of queries

        Returns:
            Rich Table object
        """
        table = Table(title=f"Conduit Router Learning Demo - Query {current_query_num}/{total_queries}")

        table.add_column("Query", style="cyan", width=40)
        table.add_column("Model", style="green", width=25)
        table.add_column("Cost", style="yellow", justify="right", width=10)
        table.add_column("Quality", style="magenta", justify="right", width=10)
        table.add_column("Complexity", style="blue", width=10)

        # Show last 10 queries
        recent_history = self.history[-10:]
        for result in recent_history:
            table.add_row(
                result["query"],
                result["model"],
                f"${result['cost']:.4f}",
                f"{result['quality']:.2f}",
                result["complexity"],
            )

        # Add summary row
        if self.history:
            total_cost = sum(r["cost"] for r in self.history)
            avg_quality = sum(r["quality"] for r in self.history) / len(self.history)
            table.add_section()
            table.add_row(
                f"[bold]Total: {len(self.history)} queries[/bold]",
                "",
                f"[bold]${total_cost:.4f}[/bold]",
                f"[bold]{avg_quality:.2f}[/bold]",
                "",
            )

        # Add router stats
        stats = self.router.hybrid_router.get_stats()
        table.add_section()
        table.add_row(
            "[bold]Router Statistics[/bold]",
            "",
            "",
            "",
            "",
        )
        table.add_row(
            f"Total Queries: {stats['total_queries']}",
            "",
            "",
            "",
            "",
        )

        # Show model selection distribution
        if "arm_pulls" in stats:
            table.add_row(
                "[bold]Model Selections:[/bold]",
                "",
                "",
                "",
                "",
            )
            for model_id, pulls in stats["arm_pulls"].items():
                percentage = (pulls / stats["total_queries"] * 100) if stats["total_queries"] > 0 else 0
                table.add_row(
                    "",
                    model_id,
                    "",
                    "",
                    f"{pulls} ({percentage:.1f}%)",
                )

        return table

    async def run(self, num_queries: int = 50, delay: float = 0.3):
        """Run the learning demo.

        Args:
            num_queries: Number of queries to simulate
            delay: Delay between queries (seconds)
        """
        self.console.print("[bold green]Conduit Router Learning Demo[/bold green]")
        self.console.print("Watch as Conduit learns to route queries optimally!\n")

        queries = self.generate_queries(num_queries)

        with Live(self.create_table(0, num_queries), refresh_per_second=4) as live:
            for i, query_info in enumerate(queries, 1):
                # Simulate query
                result = await self.simulate_query(query_info["text"], query_info["complexity"])
                self.history.append(result)

                # Update display
                live.update(self.create_table(i, num_queries))

                # Small delay for readability
                await asyncio.sleep(delay)

        # Final summary
        self.console.print("\n[bold green]Demo Complete![/bold green]\n")
        total_cost = sum(r["cost"] for r in self.history)
        avg_quality = sum(r["quality"] for r in self.history) / len(self.history)

        self.console.print(f"Total Cost: ${total_cost:.4f}")
        self.console.print(f"Average Quality: {avg_quality:.2f}")
        self.console.print(f"Total Queries: {len(self.history)}")


async def main():
    """Run the learning demo."""
    demo = LearningDemo()
    await demo.run(num_queries=50, delay=0.2)


if __name__ == "__main__":
    asyncio.run(main())

