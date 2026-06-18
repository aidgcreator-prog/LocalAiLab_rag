"""Package Agent — Export the agent configuration as a portable zip.

Inspired by the DeepAgents downloading_agents example. Creates a zip
archive containing everything needed to deploy or share this agent:

    - AGENTS.md           (shared memory / instructions)
    - subagents.yaml      (subagent definitions)
    - deepagents.toml     (project config)
    - skills/             (all skill definitions)
    - memories/           (user memory templates)
    - mcp.json            (MCP server config)
    - .env.example        (env var template, secrets stripped)

Usage:
    python package_agent.py
    python package_agent.py --output my-agent-v2.zip
    python package_agent.py --include-env   # include actual .env (careful!)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).parent

# Files to always include
CORE_FILES = [
    "AGENTS.md",
    "subagents.yaml",
    "deepagents.toml",
    "mcp.json",
]

# Directories to include recursively
CORE_DIRS = [
    "skills",
    "memories",
]

# Patterns for secrets to redact when creating .env.example
_SECRET_PATTERNS = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY)\s*=\s*.+",
    re.IGNORECASE,
)


def _make_env_example(env_path: Path) -> str:
    """Read .env and redact secret values, returning .env.example content."""
    lines: list[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        if _SECRET_PATTERNS.search(stripped):
            key = stripped.split("=", 1)[0]
            lines.append(f"{key}=<YOUR_VALUE_HERE>")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def package_agent(
    output: str | None = None,
    include_env: bool = False,
) -> Path:
    """Create the agent package zip.

    Args:
        output: Custom output filename. Defaults to lv-combined-agents-YYYYMMDD.zip
        include_env: If True, include actual .env file (use with caution)

    Returns:
        Path to the created zip file.
    """
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = f"lv-combined-agents-{stamp}.zip"

    out_path = ROOT / output

    with ZipFile(out_path, "w", ZIP_DEFLATED) as zf:
        # Core files
        for fname in CORE_FILES:
            fpath = ROOT / fname
            if fpath.exists():
                zf.write(fpath, fname)
                print(f"  + {fname}")
            else:
                print(f"  ~ {fname} (not found, skipping)")

        # Core directories
        for dirname in CORE_DIRS:
            dirpath = ROOT / dirname
            if not dirpath.is_dir():
                print(f"  ~ {dirname}/ (not found, skipping)")
                continue
            for file in sorted(dirpath.rglob("*")):
                if file.is_file():
                    arcname = file.relative_to(ROOT).as_posix()
                    zf.write(file, arcname)
                    print(f"  + {arcname}")

        # .env handling
        env_path = ROOT / ".env"
        if env_path.exists():
            if include_env:
                zf.write(env_path, ".env")
                print("  + .env (WARNING: contains secrets)")
            else:
                example = _make_env_example(env_path)
                zf.writestr(".env.example", example)
                print("  + .env.example (secrets redacted)")

    size_kb = out_path.stat().st_size / 1024
    print(f"\nPackage created: {out_path} ({size_kb:.1f} KB)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the agent configuration as a portable zip",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output zip filename (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--include-env",
        action="store_true",
        help="Include actual .env file (careful — may contain secrets!)",
    )
    args = parser.parse_args()
    package_agent(output=args.output, include_env=args.include_env)


if __name__ == "__main__":
    main()
