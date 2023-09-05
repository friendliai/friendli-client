# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPU CLI."""

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

serving_gpu_formatter = TableFormatter(
    name="Serving GPU instances",
    fields=[
        "cloud",
        "region",
        "gpu_type",
        "supported_num_gpus",
    ],
    headers=[
        "Cloud",
        "Region",
        "GPU type",
        "Supported #GPUs",
    ],
)


# pylint: disable=redefined-builtin
@app.command()
def list():
    """List up available GPUs."""
    pfs_vm_client = PFSVMClient()
    response = pfs_vm_client.list_vms()
    vm_dict = {}

    def _gpu_key(nodegroup_list_dict, nodegroup) -> str:
        return f'{nodegroup_list_dict["cloud"].upper()}-{nodegroup_list_dict["region"]}\
            -{nodegroup["vm"]["gpu_type"].upper()}'

    for nodegroup_list_dict in response:
        for nodegroup in nodegroup_list_dict["nodegroup_list"]:
            if nodegroup["vm"]["gpu_type"] in [gpu_type.value for gpu_type in GpuType]:
                gpu_key = _gpu_key(nodegroup_list_dict, nodegroup)
                if gpu_key in vm_dict:
                    vm_dict[gpu_key][
                        "supported_num_gpus"
                    ] += f', {nodegroup["vm"]["total_gpus"]}'
                else:
                    vm_dict[gpu_key] = {
                        "cloud": nodegroup_list_dict["cloud"].upper(),
                        "region": nodegroup_list_dict["region"],
                        "vm": nodegroup["vm"],
                        "gpu_type": nodegroup["vm"]["gpu_type"].upper(),
                        "supported_num_gpus": str(nodegroup["vm"]["total_gpus"]),
                    }

    serving_gpu_formatter.render(vm_dict.values())
