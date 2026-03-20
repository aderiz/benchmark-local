"""Composite widget for displaying a single metric."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Static


class MetricCard(Static):
    """Displays a metric name, value, and optional CI/CV info."""

    DEFAULT_CSS = """
    MetricCard {
        width: 1fr;
        height: auto;
        min-height: 5;
        border: solid $primary;
        padding: 1 2;
        margin: 0 1;
    }
    MetricCard .metric-label {
        color: $text-muted;
        text-style: bold;
    }
    MetricCard .metric-value {
        color: $text;
        text-style: bold;
        text-align: center;
    }
    MetricCard .metric-detail {
        color: $text-muted;
    }
    MetricCard .metric-warning {
        color: $warning;
    }
    """

    def __init__(
        self,
        label: str,
        value: str,
        detail: str = "",
        warning: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._detail = detail
        self._warning = warning

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._label, classes="metric-label")
            yield Label(self._value, classes="metric-value")
            if self._detail:
                cls = "metric-warning" if self._warning else "metric-detail"
                yield Label(self._detail, classes=cls)

    def update_value(self, value: str, detail: str = "", warning: bool = False) -> None:
        self._value = value
        self._detail = detail
        self._warning = warning
        labels = self.query(Label)
        if len(labels) >= 2:
            labels[1].update(value)
        if len(labels) >= 3:
            labels[2].update(detail)
            labels[2].set_class(warning, "metric-warning")
            labels[2].set_class(not warning, "metric-detail")
