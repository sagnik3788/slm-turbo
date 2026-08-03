from pathlib import Path
from typing import Optional

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
def serve(
    config: str,
    model: Optional[str] = typer.Option(None, help="Model to serve (defaults to the recipe's model_id)"),
    custom: bool = typer.Option(False, "--custom", help="Force the slm-turbo 4-bit KV kernel backend"),
    vllm: bool = typer.Option(False, "--vllm", help="Force stock vLLM kernels (A/B baseline)"),
    chat: bool = typer.Option(False, "--chat", help="Apply the model's chat template (needed for instruct models)"),
    prompt: str = "The history of Paris spans over two thousand years.",
    max_tokens: int = 64,
):
    if custom and vllm:
        console.print("[red]--custom and --vllm are mutually exclusive[/red]")
        raise typer.Exit(1)
    backend_override = "custom" if custom else ("vllm" if vllm else None)

    console.print(f"[bold]Serving with {config}...[/bold]")

    try:
        recipe = RecipeEngine().load(Path(config))
    except Exception as e:
        console.print(f"[red]Failed to load recipe {config}: {e}[/red]")
        raise typer.Exit(1)

    model = model or recipe.model_id
    if model is None:
        console.print(
            "[red]Recipe does not store a model_id. Re-run `slm-turbo optimize <model>` "
            "to regenerate it, or pass --model explicitly.[/red]"
        )
        raise typer.Exit(1)

    try:
        from adapters.vllm_adapter import get_available_adapter

        adapter = get_available_adapter(model, recipe, backend_override)
        adapter.start()
        metrics = adapter.get_metrics()
        console.print(
            f"[green]Engine started (attention backend: {metrics.get('backend')})[/green]"
        )
        if chat:
            text = adapter.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)[0]
        else:
            text = adapter.generate([prompt], max_tokens=max_tokens)[0]

        metrics = adapter.get_metrics()

        console.print(f"[bold]Prompt:[/bold] {prompt}")
        console.print(f"[bold]Output:[/bold] {text}")

        # ── Pretty metrics table ──
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="cyan", justify="right")
        table.add_column("Value", style="white")

        # Generation Result
        table.add_row("[bold]─ Generation Result ─[/bold]", "")
        ptok = metrics.get("prompt_tokens", 0)
        otok = metrics.get("output_tokens", 0)
        table.add_row("Tokens", f"Prompt: {ptok} | Output: {otok}")
        ttft = metrics.get("ttft_s", 0)
        tpot = metrics.get("tpot_ms", 0)
        table.add_row("Latency", f"TTFT: {ttft:.2f}s | TPOT: {tpot:.1f}ms/tok")
        thr = metrics.get("throughput_tok_s", 0)
        table.add_row("Throughput", f"{thr:.1f} tok/s (output)")
        table.add_row("Total time", f"{metrics.get('total_time_s', 0):.2f}s")

        # GPU State
        table.add_row("", "")
        table.add_row("[bold]─ GPU State ─[/bold]", "")
        gpu = metrics.get("gpu_name", "Unknown")
        table.add_row("GPU", gpu)
        vram_u = metrics.get("vram_used_mb", 0)
        vram_t = metrics.get("vram_total_mb", 0)
        vram_pct = metrics.get("vram_used_percent", 0)
        table.add_row("VRAM", f"{vram_u} / {vram_t} MB ({vram_pct}%)")
        gpu_util = metrics.get("gpu_util_percent", 0)
        table.add_row("Utilization", f"GPU: {gpu_util}%")

        # VRAM Breakdown
        mw = metrics.get("model_weights_mb", 0)
        kvu = metrics.get("kv_cache_used_mb", 0)
        eo = metrics.get("engine_overhead_mb", 0)
        if mw > 0:
            table.add_row("VRAM Breakdown", f"Weights: {mw} MB | KV: {kvu} MB | Overhead: {eo} MB")

        # Memory Bandwidth
        bw_util = metrics.get("mem_bw_util_gbps", 0)
        bw_peak = metrics.get("mem_bw_peak_gbps", 0)
        bw_pct = metrics.get("mem_bw_util_percent", 0)
        if bw_peak > 0:
            table.add_row("Memory BW", f"{bw_util} / {bw_peak} GB/s ({bw_pct}% utilized)")

        # KV Cache
        kv_used = metrics.get("kv_cache_used_tokens", 0)
        kv_total = metrics.get("kv_cache_total_tokens", 0)
        kv_pct = metrics.get("kv_cache_fill_percent", 0)
        kv_used_mb = metrics.get("kv_cache_used_mb", 0)
        kv_budget_mb = metrics.get("kv_cache_budget_mb", 0)
        if kv_total:
            table.add_row("KV Cache", f"{kv_used:,} / {kv_total:,} tokens ({kv_pct}% fill)")
            table.add_row("KV Memory", f"{kv_used_mb} / {kv_budget_mb} MB used")

        # Bytes per token comparison
        stock_bpt = metrics.get("stock_bytes_per_tok", 0)
        opt_bpt = metrics.get("opt_bytes_per_tok", 0)
        bpt_save = metrics.get("bytes_per_tok_savings", 0)
        if stock_bpt > 0:
            table.add_row("KV Bytes/Tok", f"Stock: {stock_bpt} B | Ours: {opt_bpt} B | {bpt_save}x smaller")

        # Effective context window
        remaining = metrics.get("remaining_tokens", 0)
        max_conv = metrics.get("max_conversation_tokens", 0)
        if max_conv > 0:
            table.add_row("Context Window", f"{remaining:,} tokens remaining (max: {max_conv:,})")

        # Memory efficiency
        tpg = metrics.get("throughput_per_gb", 0)
        if tpg > 0:
            table.add_row("Efficiency", f"{tpg} tok/s/GB VRAM")

        # Optimization Status
        table.add_row("", "")
        table.add_row("[bold]─ Optimization Status ─[/bold]", "")
        backend = metrics.get("backend", "vllm")
        kernel = metrics.get("kernel_serving", False)
        backend_str = f"{backend} (slm-turbo kernel)" if kernel else f"{backend} (stock)"
        table.add_row("Backend", backend_str)
        if metrics.get("kv_quant_active"):
            ratio = metrics.get("kv_quant_ratio", 4)
            table.add_row("KV Quant", f"{ratio}x compression active")
        roofline = metrics.get("roofline_decode", "unknown")
        table.add_row("Roofline", f"{roofline} (decode)")
        kcalls = metrics.get("kernel_calls", 0)
        table.add_row("Kernel calls", str(kcalls))

        console.print(table)
        adapter.stop()
    except Exception as e:
        console.print(f"[red]Serve failed: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    console.print("[bold]Live Metrics[/bold]")
    # v1 serving is one-shot: slm-turbo serve generates then exits.
    console.print(
        "[yellow]No active serving session (v1 serve is one-shot, not a daemon)[/yellow]"
    )


if __name__ == "__main__":
    app()
