"""Textual TUI application with screen routing."""

from __future__ import annotations

from textual.app import App
from textual.binding import Binding

from bench.config import BenchmarkConfig
from bench.store import SessionResult
from bench.tui.screens.config_screen import ConfigScreen
from bench.tui.screens.results_screen import ResultsScreen
from bench.tui.screens.run_screen import RunScreen


class BenchApp(App):
    """MLX Inference Benchmark TUI."""

    TITLE = "benchmark-local"
    SUB_TITLE = "MLX Inference Benchmarking for Apple Silicon"

    CSS = """
    Screen {
        background: $surface;
    }
    Header {
        dock: top;
    }
    Footer {
        dock: bottom;
    }
    .section-title {
        text-style: bold;
        color: $primary;
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    SCREENS = {
        "config": ConfigScreen,
        "run": RunScreen,
        "results": ResultsScreen,
    }

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self.session_result: SessionResult | None = None

    def on_mount(self) -> None:
        # Register screens with config
        self.install_screen(ConfigScreen(self.config), name="config")
        self.install_screen(RunScreen(self.config), name="run")
        self.install_screen(ResultsScreen(), name="results")
        self.push_screen("config")
