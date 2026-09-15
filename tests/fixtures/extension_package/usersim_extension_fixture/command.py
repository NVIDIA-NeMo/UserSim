# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def register(subparsers):
    parser = subparsers.add_parser(
        "fixture-cmd", help="Contributed by an out-of-tree package."
    )
    parser.set_defaults(func=lambda args: 0)
