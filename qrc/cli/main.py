"""CLI interface for QRC Volatility Forecaster."""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from qrc.pipeline.runner import PipelineRunner, load_config

app = typer.Typer(name="qrc", help="QRC Volatility Forecaster CLI")
console = Console()


def _run_pipeline(symbol: str, model: str, start: str | None = None) -> object:
    """Run the pipeline with progress display."""
    config = load_config()
    config["symbol"] = symbol
    config["model"] = model
    if start:
        config["start_date"] = start

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running QRC pipeline...", total=None)
        runner = PipelineRunner(config)
        ctx = runner.run()
        progress.update(task, completed=True, description="Pipeline complete")

    if ctx.errors:
        console.print(f"[red]Errors: {ctx.errors}[/red]")

    return ctx


@app.command()
def forecast(
    symbol: str = typer.Option("^GSPC", help="Yahoo Finance ticker"),
    model: str = typer.Option("QR2", help="Model: QR1 or QR2"),
    start: str = typer.Option("1990-01-01", help="Start date"),
) -> None:
    """Run volatility forecast."""
    ctx = _run_pipeline(symbol, model, start)

    if ctx.metrics is not None:
        table = Table(title=f"{model} Forecast Results")
        table.add_column("Metric")
        table.add_column("Value")
        if model in ctx.metrics.index:
            row = ctx.metrics.loc[model]
            table.add_row("MSE", f"{row['MSE']:.8f}")
            table.add_row("QLIKE", f"{row['QLIKE']:.8f}")
        console.print(table)


@app.command()
def backtest(
    symbol: str = typer.Option("^GSPC", help="Yahoo Finance ticker"),
    model: str = typer.Option("QR2", help="Model: QR1 or QR2"),
) -> None:
    """Run backtest analysis."""
    ctx = _run_pipeline(symbol, model)

    if ctx.metrics is not None:
        table = Table(title=f"{model} Backtest Summary")
        table.add_column("Metric")
        table.add_column("Value")
        if model in ctx.metrics.index:
            row = ctx.metrics.loc[model]
            table.add_row("Out-of-sample MSE", f"{row['MSE']:.8f}")
            table.add_row("Out-of-sample QLIKE", f"{row['QLIKE']:.8f}")
        console.print(table)


@app.command()
def compare(
    symbol: str = typer.Option("^GSPC", help="Yahoo Finance ticker"),
) -> None:
    """Compare all models."""
    ctx = _run_pipeline(symbol, "QR2")

    if ctx.metrics is not None:
        table = Table(title="Model Comparison")
        table.add_column("Model")
        table.add_column("MSE")
        table.add_column("QLIKE")
        table.add_column("MCS p-value")

        for name in ctx.metrics.index:
            row = ctx.metrics.loc[name]
            mcs_p = ctx.mcs_results.get(name, "-") if ctx.mcs_results else "-"
            mcs_str = f"{mcs_p:.3f}" if isinstance(mcs_p, float) else str(mcs_p)
            table.add_row(name, f"{row['MSE']:.8f}", f"{row['QLIKE']:.8f}", mcs_str)

        console.print(table)


@app.command()
def shapley(
    symbol: str = typer.Option("^GSPC", help="Yahoo Finance ticker"),
    model: str = typer.Option("QR2", help="Model: QR1 or QR2"),
) -> None:
    """Compute and display SHAP feature importance."""
    ctx = _run_pipeline(symbol, model)

    if ctx.shap_values is not None:
        mean_abs = ctx.shap_values.abs().mean().sort_values(ascending=False)
        table = Table(title="Feature Importance (Mean |SHAP|)")
        table.add_column("Feature")
        table.add_column("Mean |SHAP|")
        for feat, val in mean_abs.items():
            table.add_row(str(feat), f"{val:.6f}")
        console.print(table)

    if ctx.selected_features:
        console.print(f"\nSelected features: {ctx.selected_features}")


if __name__ == "__main__":
    app()
