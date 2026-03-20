"""Run screen: live progress + streaming output."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ProgressBar, RichLog, Static
from textual.worker import Worker, get_current_worker

from bench.config import BenchmarkConfig
from bench.runner import ProgressEvent, run_benchmark
from bench.store import SessionResult
from bench.tui.widgets.metric_card import MetricCard


class RunScreen(Screen):
    """Screen showing benchmark progress and live metrics."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    RunScreen {
        layout: vertical;
    }
    #run-status {
        height: 3;
        padding: 0 2;
        content-align: center middle;
    }
    #progress-container {
        height: 3;
        padding: 0 2;
    }
    #live-metrics {
        height: 5;
        layout: horizontal;
        padding: 0 1;
    }
    #run-log {
        height: 1fr;
        margin: 0 2;
        border: solid $primary;
    }
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self._session: SessionResult | None = None
        self._worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("Preparing benchmark...", id="run-status")
        yield ProgressBar(total=100, id="progress-bar")
        with Horizontal(id="live-metrics"):
            yield MetricCard("TTFT", "—", id="card-ttft")
            yield MetricCard("tok/s", "—", id="card-tps")
            yield MetricCard("Memory", "—", id="card-mem")
            yield MetricCard("Progress", "0%", id="card-progress")
        yield RichLog(id="run-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._worker = self.run_worker(self._run_benchmark(), exclusive=True)

    async def _run_benchmark(self) -> None:
        """Run benchmark in a worker thread."""
        log = self.query_one("#run-log", RichLog)

        def on_progress(event: ProgressEvent) -> None:
            self.call_from_thread(self._update_progress, event)

        try:
            self._session = await self.app.loop.run_in_executor(
                None,
                lambda: run_benchmark(self.config, on_progress=on_progress),
            )
            self.call_from_thread(self._on_complete)
        except Exception as e:
            self.call_from_thread(log.write, f"[red]Error: {e}[/red]")

    def _update_progress(self, event: ProgressEvent) -> None:
        """Update UI from a progress event (called on main thread)."""
        status = self.query_one("#run-status", Label)
        progress = self.query_one("#progress-bar", ProgressBar)
        log = self.query_one("#run-log", RichLog)

        pct = event.overall_progress * 100
        progress.update(progress=pct)

        card_progress = self.query_one("#card-progress", MetricCard)
        card_progress.update_value(f"{pct:.0f}%")

        if event.error:
            log.write(f"[red]{event.error}[/red]")
            return

        if event.stage == "loading":
            status.update(f"Loading {event.variant_quant} {event.family_name}...")
            log.write(f"[cyan]Loading {event.variant_repo}[/cyan]")

        elif event.stage == "warmup":
            status.update(
                f"Warmup {event.run_index}/{event.total_runs} "
                f"[{event.prompt_id}] — {event.family_name} ({event.variant_quant})"
            )

        elif event.stage == "measuring" and event.current_result:
            r = event.current_result
            status.update(
                f"Run {event.run_index}/{event.total_runs} "
                f"[{event.prompt_id}] — {event.family_name} ({event.variant_quant})"
            )
            self.query_one("#card-ttft", MetricCard).update_value(
                f"{r.ttft_ms:.1f} ms"
            )
            self.query_one("#card-tps", MetricCard).update_value(
                f"{r.tokens_per_sec:.1f}"
            )
            self.query_one("#card-mem", MetricCard).update_value(
                f"{r.peak_memory_bytes / 1024**2:.0f} MB"
            )
            log.write(
                f"  [{event.prompt_id}] "
                f"TTFT={r.ttft_ms:.1f}ms "
                f"tok/s={r.tokens_per_sec:.1f} "
                f"mem={r.peak_memory_bytes / 1024**2:.0f}MB"
            )

        elif event.stage == "quality":
            status.update(f"Quality eval — {event.message}")
            log.write(f"[yellow]{event.message}[/yellow]")

        elif event.stage == "done":
            log.write(f"[green]{event.message}[/green]")

    def _on_complete(self) -> None:
        """Called when benchmark is complete."""
        status = self.query_one("#run-status", Label)
        status.update("Benchmark complete! Press Enter to view results.")
        progress = self.query_one("#progress-bar", ProgressBar)
        progress.update(progress=100)
        if self._session is not None:
            self.app.session_result = self._session
            self.app.push_screen("results")

    def action_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.app.pop_screen()
