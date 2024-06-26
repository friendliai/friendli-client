# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Fix imports."""

from __future__ import annotations

import argparse


def adjust_imports(file_path, prefix_to_attach, imports_to_fix):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the comma-separated import targets
    imports_list = imports_to_fix.split(",")

    for import_target in imports_list:
        import_target = import_target.strip()
        old_import = f"import {import_target}"
        new_import = f"from {prefix_to_attach} import {import_target}"
        content = content.replace(old_import, new_import)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Fix import statements in generated protobuf files."
    )
    parser.add_argument("--source-path", required=True, help="The path to file to fix")
    parser.add_argument(
        "--prefix-to-attach",
        required=True,
        help="The path prefix to attach (e.g., friendli.schema.api.v1.codegen)",
    )
    parser.add_argument(
        "--imports-to-fix",
        required=True,
        help="A comma-separated string of import target to fix (e.g., response_format_pb2,completions_pb2)",
    )

    args = parser.parse_args()
    adjust_imports(args.source_path, args.prefix_to_attach, args.imports_to_fix)


if __name__ == "__main__":
    main()
