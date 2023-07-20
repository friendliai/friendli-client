# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow VM CLI."""

from __future__ import annotations

import typer

from periflow.client.deployment import PFSVMClient
from periflow.enums import GpuType
from periflow.formatter import TableFormatter

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)
quota_app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    deprecated=True,
)

serving_vm_formatter = TableFormatter(
    name="Serving VM instances",
    fields=[
        "vm.name",
        "cloud",
        "gpu_type",
        "vm.total_gpus",
        "region",
    ],
    headers=[
        "VM",
        "Cloud",
        "GPU",
        "GPU Count",
        "Region",
    ],
)


# pylint: disable=redefined-builtin
@app.command()
def list():
    """List up available VMs."""
    pfs_vm_client = PFSVMClient()
    response = pfs_vm_client.list_vms()

    vm_dict_list = [
        {
            "cloud": nodegroup_list_dict["cloud"].upper(),
            "region": nodegroup_list_dict["region"],
            "vm": nodegroup["vm"],
            "gpu_type": nodegroup["vm"]["gpu_type"].upper(),
        }
        for nodegroup_list_dict in response
        for nodegroup in nodegroup_list_dict["nodegroup_list"]
        if nodegroup["vm"]["gpu_type"] in [gpu_type.value for gpu_type in GpuType]
    ]
    serving_vm_formatter.render(vm_dict_list)
