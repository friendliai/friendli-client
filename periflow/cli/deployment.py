# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-locals, redefined-builtin, too-many-statements, too-many-arguments

"""PeriFlow Deployment CLI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID

import typer
import yaml
from dateutil.parser import parse

from periflow.client.user import UserGroupProjectClient
from periflow.enums import CloudType, DeploymentSecurityLevel, DeploymentType, GpuType
from periflow.errors import (
    AuthenticationError,
    EntityTooLargeError,
    InvalidConfigError,
    LowServicePlanError,
)
from periflow.formatter import PanelFormatter, TableFormatter
from periflow.sdk.resource.deployment import Deployment as DeploymentAPI
from periflow.utils.format import (
    datetime_to_pretty_str,
    datetime_to_simple_string,
    secho_error_and_exit,
)

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_MAX_TOKEN_COUNT = 8192

deployment_panel = PanelFormatter(
    name="Deployment Overview",
    fields=[
        "deployment_id",
        "config.name",
        "description",
        "config.deployment_type",
        "config.model_id",
        "status",
        "ready_replicas",
        "config.scaler_config.min_replica_count",
        "config.scaler_config.max_replica_count",
        "vms",
        "config.vm.gpu_type",
        "config.total_gpus",
        "start",
        "end",
        "security_level",
        "config.infrequest_log",
        "endpoint",
        "config.cloud",
        "config.region",
        "config.orca_config.max_batch_size",
        "config.orca_config.max_token_count",
        "config.orca_config.max_num_tokens_to_replace",
    ],
    headers=[
        "ID",
        "Name",
        "Description",
        "Type",
        "Ckpt ID",
        "Status",
        "#Ready",
        "Min Replicas",
        "Max Replicas",
        "VM Type",
        "GPU Type",
        "#GPUs",
        "Start",
        "End",
        "Security Level",
        "Logging",
        "Endpoint",
        "Cloud",
        "Region",
        "Max batch size",
        "Max token count",
        "Max num tokens to replace",
    ],
    extra_fields=["error"],
    extra_headers=["error"],
    substitute_exact_match_only=False,
)

deployment_table = TableFormatter(
    name="Deployments",
    fields=[
        "deployment_id",
        "config.name",
        "status",
        "ready_replicas",
        "vms",
        "config.vm.gpu_type",
        "config.total_gpus",
        "start",
        "config.cloud",
        "config.region",
    ],
    headers=[
        "ID",
        "Name",
        "Status",
        "#Ready",
        "VM Type",
        "GPU Type",
        "#GPUs",
        "Start",
        "Cloud",
        "Region",
    ],
    extra_fields=["error"],
    extra_headers=["error"],
    substitute_exact_match_only=False,
)

deployment_org_table = TableFormatter(
    name="Deployments",
    fields=[
        "deployment_id",
        "config.name",
        "status",
        "ready_replicas",
        "vms",
        "config.vm.gpu_type",
        "config.total_gpus",
        "start",
        "config.cloud",
        "config.region",
        "config.project_id",
        "project_name",
    ],
    headers=[
        "ID",
        "Name",
        "Status",
        "#Ready",
        "VM Type",
        "GPU Type",
        "#GPUs",
        "Start",
        "Cloud",
        "Region",
        "Project ID",
        "Project Name",
    ],
    extra_fields=["error"],
    extra_headers=["error"],
    substitute_exact_match_only=False,
)

deployment_metrics_table = TableFormatter(
    name="Deployment Metrics",
    fields=[
        "id",
        "latency",
        "throughput",
        "time_window",
    ],
    headers=["ID", "Latency(ms)", "Throughput(req/s)", "Time Window(sec)"],
    extra_fields=["error"],
    extra_headers=["error"],
)

deployment_usage_table = TableFormatter(
    name="Deployment Usage",
    fields=[
        "id",
        "type",
        "cloud",
        "vm",
        "created_at",
        "finished_at",
        "gpu_type",
        "duration",
    ],
    headers=[
        "ID",
        "Type",
        "Cloud",
        "VM",
        "Created At",
        "Finished At",
        "GPU",
        "Total Usage (days, HH:MM:SS)",
    ],
)

deployment_event_table = TableFormatter(
    name="Deployment Event",
    fields=[
        "id",
        "type",
        "description",
        "created_at",
    ],
    headers=["ID", "Type", "Description", "Timestamp"],
)

deployment_panel.add_substitution_rule(
    "Initializing", "[bold yellow]Initializing[/bold yellow]"
)
deployment_panel.add_substitution_rule("Healthy", "[bold green]Healthy[/bold green]")
deployment_panel.add_substitution_rule("Unhealthy", "[bold red]Unhealthy[/bold red]")
deployment_panel.add_substitution_rule(
    "Stopping", "[bold magenta]Stopping[/bold magenta]"
)
deployment_panel.add_substitution_rule("Terminated", "[bold]Terminated[/bold]")

deployment_table.add_substitution_rule(
    "Initializing", "[bold yellow]Initializing[/bold yellow]"
)
deployment_table.add_substitution_rule("Healthy", "[bold green]Healthy[/bold green]")
deployment_table.add_substitution_rule("Unhealthy", "[bold red]Unhealthy[/bold red]")
deployment_table.add_substitution_rule(
    "Stopping", "[bold magenta]Stopping[/bold magenta]"
)
deployment_table.add_substitution_rule("Terminated", "[bold]Terminated[/bold]")

deployment_org_table.add_substitution_rule(
    "Initializing", "[bold yellow]Initializing[/bold yellow]"
)
deployment_org_table.add_substitution_rule(
    "Healthy", "[bold green]Healthy[/bold green]"
)
deployment_org_table.add_substitution_rule(
    "Unhealthy", "[bold red]Unhealthy[/bold red]"
)
deployment_org_table.add_substitution_rule(
    "Stopping", "[bold magenta]Stopping[/bold magenta]"
)
deployment_org_table.add_substitution_rule("Terminated", "[bold]Terminated[/bold]")


def get_deployment_id_from_namespace(namespace: str):
    """Get deployment id from namespace."""
    return f"periflow-deployment-{namespace}"


@app.command()
def list(
    include_terminated: bool = typer.Option(
        False,
        "--include-terminated",
        help=(
            "When True, shows all deployments in my project including terminated "
            "deployments. (active deployments are shown above the terminated ones.)"
        ),
    ),
    limit: int = typer.Option(20, "--limit", help="The number of deployments to view"),
    from_oldest: bool = typer.Option(
        False, "--from-oldest", help="When True, shows from the oldest deployments."
    ),
    org: bool = typer.Option(False, "--org", help="Shows all deployments in org"),
):
    """Lists all deployments."""
    try:
        deployments = DeploymentAPI.list(
            limit=limit,
            include_terminated=include_terminated,
            from_oldest=from_oldest,
            all_in_org=org,
        )
    except AuthenticationError as exc:
        secho_error_and_exit(str(exc))

    deployment_dicts = []
    for deployment in deployments:
        deployment_dict = deployment.model_dump()
        deployment_dict["start"] = datetime_to_pretty_str(deployment.start)
        deployment_dict["vms"] = deployment.vms[0].name if deployment.vms else "None"
        deployment_dicts.append(deployment_dict)

    table = deployment_table
    if org:
        table = deployment_org_table
        project_client = UserGroupProjectClient()
        projects = project_client.list_projects()
        project_map = {project["id"]: project["name"] for project in projects}
        for deployment_dict in deployment_dicts:
            project_id = str(deployment_dict["config"]["project_id"])
            deployment_dict["project_name"] = (
                project_map[project_id] if project_id in project_map else project_id
            )

    table.render(deployment_dicts)


@app.command()
def stop(deployment_id: str = typer.Argument(..., help="ID of deployment to stop")):
    """Stops a running deployment."""
    DeploymentAPI.stop(id=deployment_id)
    typer.secho(
        f"Deployment ({deployment_id}) stopped successfully.",
        fg=typer.colors.GREEN,
    )


@app.command()
def view(
    deployment_id: str = typer.Argument(..., help="deployment id to inspect detail.")
):
    """Shows details of a deployment."""
    deployment = DeploymentAPI.get(id=deployment_id)

    deployment_dict = deployment.model_dump()
    deployment_dict["start"] = deployment.start.ctime()
    deployment_dict["end"] = deployment.end and deployment.end.ctime()
    deployment_dict["vms"] = deployment.vms[0].name if deployment.vms else None
    deployment_dict["security_level"] = (
        DeploymentSecurityLevel.PROTECTED
        if deployment.config.infrequest_perm_check
        else DeploymentSecurityLevel.PUBLIC
    )
    deployment_panel.render([deployment_dict])


@app.command()
def metrics(
    deployment_id: str = typer.Argument(
        ..., help="ID of deployment to inspect in detail."
    ),
    time_window: int = typer.Option(
        60, "--time-window", "-t", help="Time window of metrics in seconds."
    ),
):
    """Show metrics of a deployment."""
    metrics = DeploymentAPI.get_metrics(id=deployment_id, time_window=time_window)
    metrics["id"] = metrics["deployment_id"]
    if metrics["latency"]:
        # ns => ms
        metrics["latency"] = (
            f"{metrics['latency'] / 1000000:.3f}" if "latency" in metrics else None
        )
    if metrics["throughput"]:
        metrics["throughput"] = (
            f"{metrics['throughput']:.3f}" if "throughput" in metrics else None
        )
    deployment_metrics_table.render([metrics])


@app.command()
def usage(
    year: int = typer.Argument(...),
    month: int = typer.Argument(...),
    day: Optional[int] = typer.Argument(None),
):
    """Shows the usage of all deployments in a project within a specific month or day.

    :::info
    - Timestamps are recorded in UTC.
    - The usages are updated every minute.
    :::

    """
    try:
        start_date = datetime(year, month, day if day else 1, tzinfo=timezone.utc)
    except ValueError:
        secho_error_and_exit(f"Invalid date({year}-{month}{f'-{day}' if day else ''})")
    if day:
        end_date = start_date + timedelta(days=1)
    else:
        end_date = datetime(
            year + int(month == 12),
            (month + 1) if month < 12 else 1,
            1,
            tzinfo=timezone.utc,
        )
    usages = DeploymentAPI.get_usage(start_date, end_date)
    deployments = [
        {
            "id": id,
            "type": info["deployment_type"],
            "cloud": info["cloud"].upper() if "cloud" in info else None,
            "vm": info["vm"]["name"] if info.get("vm") else None,
            "gpu_type": info["vm"]["gpu_type"].upper() if info.get("vm") else None,
            "created_at": datetime_to_simple_string(parse(info["created_at"])),
            "finished_at": datetime_to_simple_string(parse(info["finished_at"]))
            if info["finished_at"]
            else "-",
            "duration": timedelta(seconds=int(info["duration"])),
        }
        for id, info in usages.items()
        if int(info["duration"]) != 0
    ]
    deployment_usage_table.render(deployments)


@app.command()
def log(
    deployment_id: str = typer.Argument(..., help="ID of deployment to get log."),
    replica_index: int = typer.Argument(
        0, help="Replica index of deployment to get log."
    ),
):
    """Shows deployments log."""
    logs = DeploymentAPI.get_logs(id=deployment_id, replica_index=replica_index)

    if len(logs) == 0:
        secho_error_and_exit("Logs are not found yet.")

    for line in logs:
        typer.echo(line["data"])


@app.command()
def create(
    checkpoint_id: UUID = typer.Option(
        ..., "--checkpoint-id", "-i", help="Checkpoint id to deploy."
    ),
    deployment_name: str = typer.Option(
        ..., "--name", "-n", help="The name of deployment. "
    ),
    cloud: CloudType = typer.Option(..., "--cloud", "-c", help="Type of cloud."),
    region: str = typer.Option(..., "--region", "-r", help="Region of cloud."),
    gpu_type: GpuType = typer.Option(
        ..., "--gpu-type", "-g", help="The GPU type for the deployment."
    ),
    num_gpus: int = typer.Option(
        ...,
        "--num-gpus",
        "-ng",
        help="The number of GPUs for the deployment. Equals to the tensor parallelism degree.",
    ),
    config_file: Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Path to configuration file."
    ),
    deployment_type: DeploymentType = typer.Option(
        DeploymentType.PROD, "--type", "-t", help="Type of deployment."
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Deployment description."
    ),
    security_level: DeploymentSecurityLevel = typer.Option(
        DeploymentSecurityLevel.PUBLIC.value,
        "--security-level",
        "-sl",
        help="Security level of deployment endpoints",
    ),
    logging: bool = typer.Option(
        False,
        "--logging",
        "-l",
        help="Logging inference requests or not.",
    ),
    default_request_config_file: Optional[typer.FileText] = typer.Option(
        None,
        "--default-request-config-file",
        "-drc",
        help="Path to default request config",
    ),
    min_replicas: int = typer.Option(
        1,
        "--min-replicas",
        "-min",
        help="Number of minimum replicas.",
    ),
    max_replicas: int = typer.Option(
        1,
        "--max-replicas",
        "-max",
        help="Number of maximum replicas.",
    ),
):
    """Creates a deployment object by using model checkpoint.

    The deployment settings are described in a configuration YAML file, and the path of that file is
    passed to the `-f` option. The following is an example YAML file:

    ```yaml
    max_batch_size: 384
    max_token_count: 12288
    max_num_tokens_to_replace: 0
    ```

    :::tip
    To turn off the deployment autoscaling, set `--min-replicas` and
    `--max-replicas` to the same value.
    :::

    :::tip
    Use `pf vm list` to find available vm-type, cloud, region, and gpu-type.
    :::

    The default request-response configuration, such as stop tokens or bad words, is
    defined in a YAML file. The path of that file is passed to the `-drc` option.
    The format of the file is as follows.

    ```json
    {
        "stop": Optional[List[str]],
        "stop_tokens": Optional[List[TokenSequence]],
        "bad_words": Optional[List[str]],
        "bad_word_tokens": Optional[List[TokenSequence]]
    }
    ```

    :::caution
    Both `bad_words` and `bad_word_tokens` cannot be set at the same time. Also, both
    `stop` and `stop_tokens` cannot be set at the same time.

    And a TokenSequence type is a dict with the key 'tokens' and the value type List[int].
    ```json
    # Wrong Example: Both `bad_words` and `bad_word_tokens` are set.
    {
        "bad_words": ["bad", "words", " bad", " words"]
        "bad_word_tokens": [{"tokens": [568, 36]}, {"tokens": [423, 76]}]
    }

    # Correct Example
    {
        "bad_words": ["bad", "words", " bad", " words"]
        "stop_tokens": [{"tokens": [568, 36]}, {"tokens": [423, 76]}]
    }
    :::

    """
    default_request_config = None
    config: Dict[str, Any] = {}
    if config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            secho_error_and_exit(
                f"Error occurred while parsing engine config file... {e}"
            )

    if "max_batch_size" not in config:
        config["max_batch_size"] = DEFAULT_MAX_BATCH_SIZE
    if "max_token_count" not in config:
        config["max_token_count"] = DEFAULT_MAX_TOKEN_COUNT

    if default_request_config_file is not None:
        try:
            default_request_config = yaml.safe_load(default_request_config_file)
        except yaml.YAMLError as e:
            secho_error_and_exit(
                f"Error occurred while parsing default request config file... {e}"
            )

    try:
        deployment = DeploymentAPI.create(
            checkpoint_id=checkpoint_id,
            name=deployment_name,
            deployment_type=deployment_type,
            cloud=cloud,
            region=region,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            config=config,
            description=description,
            default_request_config=default_request_config,
            security_level=security_level,
            logging=logging,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
    except (
        AuthenticationError,
        InvalidConfigError,
        EntityTooLargeError,
        LowServicePlanError,
    ) as exc:
        secho_error_and_exit(str(exc))

    typer.secho(
        f"Deployment ({deployment.deployment_id}) started successfully. "
        f"Use 'pf deployment view {deployment.deployment_id}' to see the deployment details.\n"
        f"Send inference requests to '{deployment.endpoint}'.",
        fg=typer.colors.GREEN,
    )


@app.command()
def update(
    deployment_id: str = typer.Argument(..., help="ID of deployment to update."),
    min_replicas: int = typer.Option(
        ..., "--min-replicas", "-min", help="Set min_replicas of deployment."
    ),
    max_replicas: int = typer.Option(
        ..., "--max-replicas", "-max", help="Set max_replicas of deployment."
    ),
):
    """Updates configuration of a deployment.

    :::tip
    To turn off the deployment autoscaling, set `--min-replicas` and
    `--max-replicas` to the same value.
    :::

    """
    DeploymentAPI.adjust_replica_config(
        id=deployment_id, min_replicas=min_replicas, max_replicas=max_replicas
    )
    typer.secho(
        f"Scaler of deployment ({deployment_id}) is updated.\n"
        f"Set min_replicas to {min_replicas}, max_replicas to {max_replicas}",
        fg=typer.colors.GREEN,
    )


@app.command()
def event(
    deployment_id: str = typer.Argument(..., help="ID of deployment to get events."),
):
    """Gets deployment events."""
    events = DeploymentAPI.get_events(id=deployment_id)
    for event in events:
        event["id"] = f"periflow-deployment-{event['namespace']}"
        event["created_at"] = datetime_to_simple_string(parse(event["created_at"]))
    deployment_event_table.render(events)


@app.command()
def req_resp(
    deployment_id: str = typer.Argument(
        ..., help="ID of deployment to download request-response logs."
    ),
    since: str = typer.Option(
        ...,
        "--since",
        help=(
            "Start time of request-response logs. The format should be {YYYY}-{MM}-{DD}T{HH}. "
            "The UTC timezone will be used by default."
        ),
    ),
    until: str = typer.Option(
        ...,
        "--until",
        help=(
            "End time of request-response logs. The format should be {YYYY}-{MM}-{DD}T{HH}. "
            "The UTC timezone will be used by default."
        ),
    ),
    save_dir: Optional[str] = typer.Option(
        None,
        "-s",
        "--save-dir",
        help="Directory path to save request-response logs",
    ),
):
    """Downloads request-response logs for a deployment."""
    try:
        start = datetime.strptime(since, "%Y-%m-%dT%H")
        end = datetime.strptime(until, "%Y-%m-%dT%H")
    except ValueError:
        secho_error_and_exit(
            "Invalid datetime format. The format should be {YYYY}-{MM}-{DD}T{HH} "
            "(e.g., 1999-01-01T01)."
        )

    DeploymentAPI.download_req_resp_logs(
        id=deployment_id, since=start, until=end, save_dir=save_dir
    )
