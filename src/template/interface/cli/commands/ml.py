from typing import Optional

import typer
from tqdm import tqdm

from template.interface.cli.commands.dataset import get_service_dataset
from template.interface.cli.commands.tokenizer import get_service_tokenizer
from template.core.cli import AsyncTyper

cli_ml = AsyncTyper(
    name="ml",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Commands for creating and managing models.",
)


async def get_service_ml():  # noqa
    """Get the tokenizer service."""

    from template.infrastructure.repositories.ml import MLRepository
    from template.application.ml import MLService
    from template.settings import get_settings

    from template.settings import get_storage_infra

    settings = get_settings()

    infra_client = get_storage_infra(settings)
    repo = MLRepository(infra_client=infra_client)
    return MLService(repo_ml=repo)


@cli_ml.command(name="train")
async def train_model(
    model_id: str = typer.Argument("model", help="Model identifier to train"),
    dataframe: str = typer.Option("villes", "--dataset", help="Dataset identifier to use to train the model"),
    tokenizer: str = typer.Option("tokenizer", "--tokenizer", help="Tokenizer identifier to use to train the model"),
    device: str = "cuda",
    batch_size: int = 256,
    ratio_tests: float = 0.1,
    ratio_validation: float = 0.1,
    d_model: int = 256,
    d_hidden: int = 256,
    n_context: int = 10,
    lr: float = 1e-3,
    num_epochs: int = 20,
    scheduler_start_factor: float = 1.0,
    scheduler_end_factor: float = 1e-4,
    scheduler_total_iters: int = 0,
):
    """Create a model from a dataset and a tokenizer."""
    service_ml = await get_service_ml()
    service_dataset = await get_service_dataset()
    service_tokenizer = await get_service_tokenizer()

    dataframe = await service_dataset.get(identifier=dataframe)
    tokenizer = await service_tokenizer.get(identifier=tokenizer)

    typer.echo("Training… (Ctrl-C to abort)")
    train_config = {
        "device": device,
        "batch_size": batch_size,
        "ratio_tests": ratio_tests,
        "ratio_validation": ratio_validation,
        "d_model": d_model,
        "d_hidden": d_hidden,
        "n_context": n_context,
        "lr": lr,
        "num_epochs": num_epochs,
        "scheduler_start_factor": scheduler_start_factor,
        "scheduler_end_factor": scheduler_end_factor,
        "scheduler_total_iters": scheduler_total_iters,
    }
    model = await service_ml.train(
        id_=model_id,
        dataframe=dataframe,
        tokenizer=tokenizer,
        **train_config,
    )
    typer.echo(f"Model '{model_id}' saved (vocab size = {len(model.tokenizer.vocab)}).")


@cli_ml.command(name="create")
async def create_model(
    model_id: str = typer.Argument("model", help="Model identifier to train"),
    tokenizer_id: str = typer.Option("tokenizer", "--tokenizer", help="Tokenizer identifier to use to train the model"),
    d_model: int = 256,
    d_hidden: int = 256,
    n_context: int = 10,
):
    """Delete a model from the repository."""

    service_ml = await get_service_ml()
    service_tokenizer = await get_service_tokenizer()

    tokenizer_id = await service_tokenizer.get(identifier=tokenizer_id)

    await service_ml.create(
        model_id=model_id,
        tokenizer=tokenizer_id,
        d_model=d_model,
        d_hidden=d_hidden,
        n_context=n_context,
    )
    typer.echo(f"Model '{model_id}' deleted successfully.")


@cli_ml.command(name="delete")
async def delete_model(identifier: str = typer.Argument(..., help="Model identifier to delete")):
    """Delete a model from the repository."""

    service_ml = await get_service_ml()
    await service_ml.delete(id_=identifier)
    typer.echo(f"Model '{identifier}' deleted successfully.")


@cli_ml.command(name="list")
async def list_models():
    """List available models from the repository."""
    service_ml = await get_service_ml()

    models = await service_ml.list()
    if not models:
        typer.echo("No models found.")
    else:
        typer.echo("Available models:")
        for m in models:
            typer.echo(f"- {m}")


@cli_ml.command(name="generate")
async def generate_text(
    model_id: str = typer.Argument("model", help="Model identifier to use"),
    start_text: str = typer.Option("", "--start-text", help="Start generation from this text"),
    max_length: int = typer.Option(64, "--max-length", help="Maximum length of generated text"),
    num_samples: int = typer.Option(10, "--num-samples", help="Number of samples to generate"),
    device: str = typer.Option("cuda", "--device", help="Device to use for generation"),
    temperature: float = typer.Option(1.0, "--temperature", help="Temperature for sampling"),
    top_p: float = typer.Option(1.0, "--top-p", help="Nucleus sampling threshold"),
    n: int = typer.Option(10, "--n", help="Number of samples to generate (deprecated, use --num-samples)"),
):
    """Generate text from a model."""
    service_ml = await get_service_ml()

    model = await service_ml.get(identifier=model_id)
    model = model.to(device)
    model.eval()

    samples = set()
    for _ in tqdm(range(n), desc="Generating"):
        result = model.generate_city_name(
            start_tokens=start_text,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        samples.add(result)
        if len(samples) >= num_samples:
            break

    typer.echo(f"Generated {len(samples)} samples:")
    for s in samples:
        typer.echo(f"- {s}")


@cli_ml.command(name="prob")
async def show_probabilities(
    identifier: str = typer.Argument("model", help="Model identifier to use"),
    start_text: str = typer.Option("", "--start-text", help="Check probabilities from this text"),
    device: str = typer.Option("cuda", "--device", help="Device to use for generation"),
    temperature: float = typer.Option(1.0, "--temperature", help="Temperature for sampling"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Nucleus sampling threshold"),
):
    """Show token probabilities for next-token prediction."""
    import torch
    from torch.nn import functional as F
    from rich import print as rprint

    service_ml = await get_service_ml()

    model = await service_ml.get(identifier=identifier)
    tokenizer = model.tokenizer
    model.eval()

    tokens = tokenizer.encode(start_text)
    ids = [tokenizer.sos_index] + tokens
    x = torch.tensor([ids], device=device)
    if x.shape[1] > model.n_context:
        x = x[:, -model.n_context :]

    with torch.no_grad():
        logits = model(x)[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        top_probs, top_idx = torch.sort(probs, descending=True)

    mask = torch.ones_like(top_probs, dtype=torch.bool)
    if top_p is not None and top_p < 1.0:
        cum = torch.cumsum(top_probs, dim=0)
        mask = cum <= top_p
        if not mask.any():
            mask[0] = True

    rprint(f"\n[bold]Top-20 token probabilities after '{start_text}':[/bold]")
    for i in range(20):
        token, prob = tokenizer.vocab[top_idx[i].item()], top_probs[i].item()
        color = "green" if mask[i] else "white"
        rprint(f"[{color}]{token:<15} {prob:.4f}[/{color}]")
