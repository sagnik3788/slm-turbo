from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from optimizers import optimizers
from profiler import device_info, profile_analytical
from recipe import RecipeEngine
from roofline import classify_bottleneck
from topology import build_topology

app = typer.Typer()
console = Console()


@app.command()
def analyze(model: str):
    console.print(f"[bold]Analyzing {model}...[/bold]")

    # Probe
    topology = build_topology(model)
    device = device_info()
    snapshot = profile_analytical(topology)
    bottleneck = classify_bottleneck(snapshot, device)

    # Print table
    table = Table(title="Diagnosis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model", topology.model_id)
    table.add_row("Params", f"{snapshot.model_params / 1e9:.1f}B")
    table.add_row("GPU", f"{device.name} ({device.vram_mb}MB)")
    table.add_row("Prefill", bottleneck.prefill)
    table.add_row("Decode", bottleneck.decode)

    console.print(table)


@app.command()
def optimize(model: str):
    console.print(f"[bold]Optimizing {model}...[/bold]")

    # Full pipeline
    topology = build_topology(model)
    device = device_info()
    snapshot = profile_analytical(topology)
    bottleneck = classify_bottleneck(snapshot, device)
    steps = optimizers(device, topology, snapshot, bottleneck)

    # Generate recipe
    engine = RecipeEngine()
    recipe = engine.generate(topology, device, steps)

    recipe_dir = Path.home() / ".slm-turbo" / "recipes"
    recipe_path = (
        recipe_dir
        / f"{topology.model_id.replace('/', '_')}_{recipe.device_hash[:8]}.yaml"
    )
    engine.save(recipe, recipe_path)

    console.print(f"[green]Recipe saved to {recipe_path}[/green]")
    console.print(f"Steps: {[s.name for s in steps]}")


@app.command()
def serve(model: str, config: str):
    console.print(f"[bold]Serving {model} with {config}...[/bold]")
    # TODO: Load recipe → start adapter → print URL
    console.print("[yellow]Adapter not yet implemented[/yellow]")


@app.command()
def status():
    console.print("[bold]Live Metrics[/bold]")
    # TODO: Read from adapter
    console.print("[yellow]Not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
