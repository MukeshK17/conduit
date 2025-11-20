"""Unit tests for CLI commands."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import BaseModel

from conduit.cli.main import cli, demo, run, serve, version


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestCLIEntryPoint:
    """Tests for CLI entry point."""

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Conduit - ML-powered LLM routing system" in result.output
        assert "serve" in result.output
        assert "run" in result.output
        assert "demo" in result.output
        assert "version" in result.output

    def test_cli_no_command(self, runner):
        """Test CLI with no command shows help."""
        result = runner.invoke(cli, [])
        # Click returns exit code 2 for missing command
        assert result.exit_code in [0, 2]


class TestServeCommand:
    """Tests for serve command."""

    @patch("conduit.cli.main.uvicorn.run")
    @patch("conduit.cli.main.create_app")
    def test_serve_defaults(self, mock_create_app, mock_uvicorn_run, runner):
        """Test serve command with default options."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        result = runner.invoke(serve, [])

        assert result.exit_code == 0
        mock_create_app.assert_called_once()
        mock_uvicorn_run.assert_called_once()
        call_args = mock_uvicorn_run.call_args

        # Verify uvicorn called with app
        assert call_args[0][0] == mock_app

    @patch("conduit.cli.main.uvicorn.run")
    @patch("conduit.cli.main.create_app")
    def test_serve_custom_host_port(self, mock_create_app, mock_uvicorn_run, runner):
        """Test serve with custom host and port."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        result = runner.invoke(serve, ["--host", "0.0.0.0", "--port", "9000"])

        assert result.exit_code == 0
        call_args = mock_uvicorn_run.call_args
        assert call_args[1]["host"] == "0.0.0.0"
        assert call_args[1]["port"] == 9000

    @patch("conduit.cli.main.uvicorn.run")
    @patch("conduit.cli.main.create_app")
    def test_serve_with_reload(self, mock_create_app, mock_uvicorn_run, runner):
        """Test serve with reload flag."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        result = runner.invoke(serve, ["--reload"])

        assert result.exit_code == 0
        call_args = mock_uvicorn_run.call_args
        assert call_args[1]["reload"] is True

    @patch("conduit.cli.main.uvicorn.run")
    @patch("conduit.cli.main.create_app")
    def test_serve_log_level(self, mock_create_app, mock_uvicorn_run, runner):
        """Test serve with custom log level."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        result = runner.invoke(serve, ["--log-level", "debug"])

        assert result.exit_code == 0
        call_args = mock_uvicorn_run.call_args
        assert call_args[1]["log_level"] == "debug"


class TestRunCommand:
    """Tests for run command."""

    @pytest.fixture
    def mock_service(self):
        """Create mock service with async methods."""

        class SimpleResult(BaseModel):
            content: str

        mock_result = MagicMock()
        mock_result.id = "result_123"
        mock_result.query_id = "query_456"
        mock_result.model = "openai:gpt-4o-mini"
        mock_result.data = {"content": "2+2 equals 4"}
        mock_result.metadata = {
            "routing_confidence": 0.95,
            "cost": 0.0001,
            "latency": 1.2,
            "tokens": 50,
            "reasoning": "Selected gpt-4o-mini for simple math",
        }

        service = MagicMock()
        service.complete = AsyncMock(return_value=mock_result)
        service.database.disconnect = AsyncMock()

        return service

    @patch("conduit.cli.main.create_service")
    def test_run_basic_query(self, mock_create_service, mock_service, runner):
        """Test run command with basic query."""
        # create_service is async, so return the mock service directly
        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(run, ["--query", "What is 2+2?"])

        assert result.exit_code == 0
        assert "Routing Results" in result.output
        assert "openai:gpt-4o-mini" in result.output
        assert "Confidence: 0.95" in result.output
        assert "Cost: $0.000100" in result.output
        assert "Latency: 1.20s" in result.output
        assert "2+2 equals 4" in result.output

    @patch("conduit.cli.main.create_service")
    def test_run_json_output(self, mock_create_service, mock_service, runner):
        """Test run command with JSON output."""
        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(run, ["--query", "Test query", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["id"] == "result_123"
        assert output["query_id"] == "query_456"
        assert output["model"] == "openai:gpt-4o-mini"
        assert output["data"]["content"] == "2+2 equals 4"

    @patch("conduit.cli.main.create_service")
    def test_run_with_constraints(self, mock_create_service, mock_service, runner):
        """Test run with cost/latency/quality constraints."""
        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(
            run,
            [
                "--query",
                "Complex query",
                "--max-cost",
                "0.01",
                "--max-latency",
                "5.0",
                "--min-quality",
                "0.8",
            ],
        )

        assert result.exit_code == 0
        # Verify service.complete was called with constraints
        mock_service.complete.assert_called_once()
        call_args = mock_service.complete.call_args
        constraints = call_args[1].get("constraints")
        assert constraints is not None
        assert constraints["max_cost"] == 0.01
        assert constraints["max_latency"] == 5.0
        assert constraints["min_quality"] == 0.8

    @patch("conduit.cli.main.create_service")
    def test_run_with_provider(self, mock_create_service, mock_service, runner):
        """Test run with preferred provider."""
        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(
            run, ["--query", "Test", "--provider", "anthropic", "--user-id", "user_123"]
        )

        assert result.exit_code == 0
        call_args = mock_service.complete.call_args
        constraints = call_args[1].get("constraints")
        assert constraints["preferred_provider"] == "anthropic"
        assert call_args[1].get("user_id") == "user_123"

    @patch("conduit.cli.main.create_service")
    def test_run_error_handling(self, mock_create_service, runner):
        """Test run command error handling."""
        mock_service = MagicMock()
        mock_service.complete = AsyncMock(
            side_effect=Exception("API connection failed")
        )
        mock_service.database.disconnect = AsyncMock()

        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(run, ["--query", "Test query"])

        assert result.exit_code == 1
        assert "Error: API connection failed" in result.output


class TestDemoCommand:
    """Tests for demo command."""

    @pytest.fixture
    def mock_service(self):
        """Create mock service for demo."""

        mock_result = MagicMock()
        mock_result.model = "openai:gpt-4o-mini"
        mock_result.metadata = {"cost": 0.0001, "latency": 1.2}

        service = MagicMock()
        service.complete = AsyncMock(return_value=mock_result)
        service.database.disconnect = AsyncMock()

        return service

    @patch("conduit.cli.main.create_service")
    def test_demo_default_queries(self, mock_create_service, mock_service, runner):
        """Test demo with default number of queries."""
        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(demo, [])

        assert result.exit_code == 0
        assert "Conduit Demo - ML-Powered LLM Routing" in result.output
        assert "Running 10 demo queries" in result.output
        assert "Demo Summary" in result.output
        assert "Total Queries: 10" in result.output
        assert "Model Distribution:" in result.output

    @patch("conduit.cli.main.create_service")
    def test_demo_custom_query_count(self, mock_create_service, mock_service, runner):
        """Test demo with custom query count."""
        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(demo, ["--queries", "5"])

        assert result.exit_code == 0
        assert "Running 5 demo queries" in result.output
        assert "Total Queries: 5" in result.output

    @patch("conduit.cli.main.create_service")
    def test_demo_with_compare(self, mock_create_service, mock_service, runner):
        """Test demo with comparison flag."""
        async def mock_factory(*args, **kwargs):
            return mock_service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(demo, ["--queries", "3", "--compare"])

        assert result.exit_code == 0
        assert "Comparison with Static Routing" in result.output

    @patch("conduit.cli.main.create_service")
    def test_demo_partial_failures(self, mock_create_service, runner):
        """Test demo with some query failures."""
        mock_result = MagicMock()
        mock_result.model = "openai:gpt-4o-mini"
        mock_result.metadata = {"cost": 0.0001, "latency": 1.2}

        service = MagicMock()
        # Fail on second query
        service.complete = AsyncMock(
            side_effect=[
                mock_result,
                Exception("Timeout"),
                mock_result,
            ]
        )
        service.database.disconnect = AsyncMock()

        async def mock_factory(*args, **kwargs):
            return service

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(demo, ["--queries", "3"])

        assert result.exit_code == 0
        assert "✗ Error: Timeout" in result.output
        assert "Demo Summary" in result.output

    @patch("conduit.cli.main.create_service")
    def test_demo_complete_failure(self, mock_create_service, runner):
        """Test demo with service creation failure."""

        async def mock_factory(*args, **kwargs):
            raise Exception("Database connection failed")

        mock_create_service.side_effect = mock_factory

        result = runner.invoke(demo, [])

        assert result.exit_code == 1
        assert "Error: Database connection failed" in result.output


class TestVersionCommand:
    """Tests for version command."""

    def test_version_output(self, runner):
        """Test version command output."""
        result = runner.invoke(version, [])

        assert result.exit_code == 0
        assert "Conduit v0.1.0" in result.output


class TestCommandValidation:
    """Tests for command option validation."""

    def test_run_missing_query(self, runner):
        """Test run command fails without query."""
        result = runner.invoke(run, [])

        assert result.exit_code != 0
        assert "Error" in result.output or "Missing option" in result.output

    def test_serve_invalid_log_level(self, runner):
        """Test serve with invalid log level."""
        result = runner.invoke(serve, ["--log-level", "invalid"])

        assert result.exit_code != 0

    def test_demo_invalid_queries(self, runner):
        """Test demo with invalid query count."""
        result = runner.invoke(demo, ["--queries", "invalid"])

        assert result.exit_code != 0
