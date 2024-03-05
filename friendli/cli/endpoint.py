# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-locals, redefined-builtin, too-many-statements, too-many-arguments

"""Friendli Endpoint CLI."""

from __future__ import annotations

import typer

from friendli.formatter import PanelFormatter, TableFormatter
from friendli.sdk.client import Friendli
from friendli.utils.compat import model_dump
from friendli.utils.decorator import check_api

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)

DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_MAX_TOKEN_COUNT = 8192

panel_formatter = PanelFormatter(
    name="Endpoint Overview",
    fields=[
        "id",
        "name",
        "hfModelRepo",
        "gpuType",
        "numGpu",
        "status",
        "phase.step",
        "endpointUrl",
        "phase.currReplica",
        "phase.desiredReplica",
        "createdBy.email",
        "createdAt",
        "updatedAt",
    ],
    headers=[
        "ID",
        "Name",
        "Model",
        "GPU Type",
        "GPU Count",
        "Status",
        "Detailed Status",
        "Endpoint URL",
        "Current # Replicas",
        "Desired # Replicas",
        "Created By",
        "Created At",
        "Updated At",
    ],
    substitute_exact_match_only=False,
)

table_formatter = TableFormatter(
    name="Endpoints",
    fields=[
        "id",
        "name",
        "hfModelRepo",
        "gpuType",
        "numGpu",
        "status",
        "createdAt",
    ],
    headers=[
        "ID",
        "Name",
        "Model",
        "GPU Type",
        "GPU Count",
        "Status",
        "Created At",
    ],
    substitute_exact_match_only=False,
)

panel_formatter.add_substitution_rule(
    "INITIALIZING", "[bold yellow]INITIALIZING[/bold yellow]"
)
panel_formatter.add_substitution_rule("RUNNING", "[bold green]RUNNING[/bold green]")
panel_formatter.add_substitution_rule("SLEEPING", "[bold cyan]SLEEPING[/bold cyan]")
panel_formatter.add_substitution_rule(
    "STOPPING", "[bold magenta]STOPPING[/bold magenta]"
)
panel_formatter.add_substitution_rule("FAILED", "[bold red]FAILED[/bold red]")
panel_formatter.add_substitution_rule("TERMINATED", "[bold]TERMINATED[/bold]")

table_formatter.add_substitution_rule(
    "INITIALIZING", "[bold yellow]INITIALIZING[/bold yellow]"
)
table_formatter.add_substitution_rule("RUNNING", "[bold green]RUNNING[/bold green]")
table_formatter.add_substitution_rule("SLEEPING", "[bold cyan]SLEEPING[/bold cyan]")
table_formatter.add_substitution_rule(
    "STOPPING", "[bold magenta]STOPPING[/bold magenta]"
)
table_formatter.add_substitution_rule("FAILED", "[bold red]FAILED[/bold red]")
table_formatter.add_substitution_rule("TERMINATED", "[bold]TERMINATED[/bold]")


@app.command()
@check_api
def create(
    name: str = typer.Option(
        ..., "--name", "-n", help="The name of endpoint to create."
    ),
    model_repo: str = typer.Option(
        ..., "--model", "-m", help="The name of Hugging Face model to deploy."
    ),
    gpu_type: str = typer.Option(
        ..., "--gpu-type", "-gt", help="GPU type to serve the deployed model."
    ),
    gpu_count: int = typer.Option(..., "--gpu-count", "-gc"),
):
    """Creates a new endpoint with deploying model."""
    client = Friendli()

    endpoint = client.endpoint.create(
        name=name,
        model_repo=model_repo,
        gpu_type=gpu_type,
        num_gpus=gpu_count,
    )
    panel_formatter.render(model_dump(endpoint))


@app.command("list")
@check_api
def list_endpoints():
    """List endpoints."""
    client = Friendli()
    endpoints = client.endpoint.list()
    endpoints_ = [model_dump(endpoint) for endpoint in iter(endpoints)]
    table_formatter.render(endpoints_)


@app.command()
@check_api
def get(endpoint_id: str = typer.Argument(..., help="ID of an endpoint to get.")):
    """Get a detailed info of an endpoint."""
    client = Friendli()
    endpoint = client.endpoint.get(endpoint_id)
    panel_formatter.render(model_dump(endpoint))


@app.command()
@check_api
def terminate(
    endpoint_id: str = typer.Argument(..., help="ID of an endpoint to terminate.")
):
    """Terminate a running endpoint."""
    client = Friendli()
    client.endpoint.terminate(endpoint_id=endpoint_id)
    typer.secho(
        f"Endpoint '{endpoint_id}' is terminated successfully.", fg=typer.colors.GREEN
    )
