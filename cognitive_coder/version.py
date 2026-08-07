# SPDX-License-Identifier: Apache-2.0
"""The single source of truth for the version.

Read from HERE, never from package metadata. A host that vendors this repo as
a git submodule and inserts it on ``sys.path`` has no installed distribution,
so ``importlib.metadata.version("cognitive-coder")`` raises for them — and the
vendoring path is a supported one (§1.2, M50). One module-level string costs
nothing and works in every case.
"""

from __future__ import annotations

__version__ = "0.9.0"

# The API surface frozen at 1.0 is `cognitive_coder/__init__.py` (C9). Until
# then this is pre-release and the Ports may still move; after 1.0 a breaking
# change to a Port OR to a shared type in types.py is a major version bump.
API_VERSION = "1.0-draft"

# Phases of the build spec that are implemented in this tree. Reported by
# `ccoder doctor` so nobody has to guess what is and is not here yet.
IMPLEMENTED_PHASES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
