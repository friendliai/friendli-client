# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-locals, redefined-builtin, too-many-statements, too-many-arguments

"""Friendli Deployment CLI."""

from __future__ import annotations

import typer
import yaml

from friendli.formatter import PanelFormatter, TableFormatter
from friendli.schema.config.deployment import DeploymentConfig
from friendli.sdk.client import Friendli
from friendli.utils.compat import model_dump, model_parse
from friendli.utils.decorator import check_api

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)

DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_MAX_TOKEN_COUNT = 8192

deployment_panel = PanelFormatter(
    name="Deployment Overview",
    fields=[
        "id",
        "name",
        "gpuType",
        "numGpu",
        "status",
        "createdAt",
        "updatedAt",
    ],
    headers=[
        "ID",
        "Name",
        "GPU Type",
        "GPU Count",
        "Status",
        "Created At",
        "Updated At",
    ],
    substitute_exact_match_only=False,
)

deployment_table = TableFormatter(
    name="Deployments",
    fields=[
        "id",
        "name",
        "gpuType",
        "numGpu",
        "status",
        "createdAt",
        "updatedAt",
    ],
    headers=[
        "ID",
        "Name",
        "GPU Type",
        "GPU Count",
        "Status",
        "Created At",
        "Updated At",
    ],
    substitute_exact_match_only=False,
)

deployment_panel.add_substitution_rule(
    "Initializing", "[bold yellow]Initializing[/bold yellow]"
)
deployment_panel.add_substitution_rule("Healthy", "[bold green]Healthy[/bold green]")
deployment_panel.add_substitution_rule("Unhealthy", "[bold red]Unhealthy[/bold red]")
deployment_panel.add_substitution_rule(
    "Stopping", "[bold magenta]Stopping[/bold magenta]"
)
deployment_panel.add_substitution_rule("Failed", "[bold red]Failed[/bold red]")
deployment_panel.add_substitution_rule("Terminated", "[bold]Terminated[/bold]")

deployment_table.add_substitution_rule(
    "Initializing", "[bold yellow]Initializing[/bold yellow]"
)
deployment_table.add_substitution_rule("Healthy", "[bold green]Healthy[/bold green]")
deployment_table.add_substitution_rule("Unhealthy", "[bold red]Unhealthy[/bold red]")
deployment_table.add_substitution_rule(
    "Stopping", "[bold magenta]Stopping[/bold magenta]"
)
deployment_table.add_substitution_rule("Failed", "[bold red]Failed[/bold red]")
deployment_table.add_substitution_rule("Terminated", "[bold]Terminated[/bold]")


@app.command()
@check_api
def create(
    config_file: typer.FileText = typer.Option(
        None,
        "-f",
        "--config-file",
        help="Path to deployment config file.",
    )
):
    """Creates a deployment object by using model checkpoint."""
    client = Friendli()
    config_dict = yaml.safe_load(config_file)
    config = model_parse(DeploymentConfig, config_dict)

    deployment = client.deployment.create(
        name=config.name,
        gpu_type=config.gpu.type,
        num_gpus=config.gpu.count,
        backbone_eid=config.model,
        adapter_eids=config.adapters,
        launch_config=config.launch_config,
    )
    deployment_panel.render(model_dump(deployment))


@app.command("list")
@check_api
def list_deployments():
    """List deployments."""
    client = Friendli()
    deployments = client.deployment.list()
    deployments_ = [model_dump(deployment) for deployment in iter(deployments)]
    deployment_table.render(deployments_)
