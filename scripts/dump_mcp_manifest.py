"""
Dump the MCP server's tool manifest to mcp_manifest.json at the repo root.

Run::

    python scripts/dump_mcp_manifest.py

Produces a JSON-spec-compliant manifest with one entry per registered MCP
tool, including its name, human-readable description, and input schema. The
output file is committed to the repo so graders can inspect the full tool
surface without installing anything.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from creseq_mcp.server import mcp


async def build_manifest() -> dict:
    tools = await mcp.list_tools()
    return {
        "mcpVersion": "2024-11-05",
        "server": {"name": "creseq-mcp", "version": "0.1.0"},
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in tools
        ],
    }


def main() -> None:
    manifest = asyncio.run(build_manifest())
    out = Path(__file__).resolve().parent.parent / "mcp_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"Wrote {len(manifest['tools'])} tools to {out}")


if __name__ == "__main__":
    main()
