"""Configuration screen: model/param selection."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Static,
)

from bench.config import BenchmarkConfig, ModelFamily


class ConfigScreen(Screen):
    """Screen for selecting models and benchmark parameters."""

    BINDINGS = [("escape", "app.quit", "Quit")]

    DEFAULT_CSS = """
    ConfigScreen {
        layout: vertical;
    }
    #config-params {
        height: auto;
        padding: 1 2;
        border: solid $primary;
        margin: 1 2;
    }
    #config-params Label {
        margin: 0 1;
    }
    #config-params Input {
        width: 12;
        margin: 0 1;
    }
    .param-row {
        height: 3;
        layout: horizontal;
    }
    #model-list {
        height: 1fr;
        padding: 1 2;
        margin: 0 2;
    }
    .family-row {
        height: 3;
        layout: horizontal;
    }
    .family-row Label {
        width: 1fr;
        content-align: left middle;
    }
    .family-row Checkbox {
        width: auto;
    }
    #start-btn {
        dock: bottom;
        width: 100%;
        margin: 1 2;
    }
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self._family_checkboxes: dict[str, Checkbox] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="config-params"):
            yield Label("Benchmark Parameters", classes="section-title")

            with Horizontal(classes="param-row"):
                yield Label("Warmup runs:")
                yield Input(
                    str(self.config.warmup_runs),
                    id="warmup-input",
                    type="integer",
                )
                yield Label("Measured runs:")
                yield Input(
                    str(self.config.measured_runs),
                    id="measured-input",
                    type="integer",
                )

            with Horizontal(classes="param-row"):
                yield Label("Max tokens:")
                yield Input(
                    str(self.config.max_tokens),
                    id="max-tokens-input",
                    type="integer",
                )
                yield Label("Temperature:")
                yield Input(
                    str(self.config.temperature),
                    id="temp-input",
                    type="number",
                )

        with VerticalScroll(id="model-list"):
            yield Label("Model Families (select which to benchmark)")
            for family in self.config.model_families:
                cb = Checkbox(
                    f"{family.name} ({family.size}, {family.kind}, "
                    f"{len(family.variants)} variants)",
                    value=True,
                    id=f"fam-{family.name.replace(' ', '-')}",
                )
                self._family_checkboxes[family.name] = cb
                yield cb

        yield Button("Start Benchmark", id="start-btn", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            self._apply_config()
            self.app.push_screen("run")

    def _apply_config(self) -> None:
        """Apply UI inputs back to config."""
        warmup = self.query_one("#warmup-input", Input)
        measured = self.query_one("#measured-input", Input)
        max_tokens = self.query_one("#max-tokens-input", Input)
        temp = self.query_one("#temp-input", Input)

        try:
            self.config.warmup_runs = int(warmup.value)
        except ValueError:
            pass
        try:
            self.config.measured_runs = int(measured.value)
        except ValueError:
            pass
        try:
            self.config.max_tokens = int(max_tokens.value)
        except ValueError:
            pass
        try:
            self.config.temperature = float(temp.value)
        except ValueError:
            pass

        # Filter out unchecked families
        self.config.model_families = [
            f for f in self.config.model_families
            if self._family_checkboxes.get(f.name, Checkbox()).value
        ]
