"""Real tools — they touch the actual filesystem/git/pytest, nothing is
simulated. Triggered by slash-commands in coconut_tui/agent.py."""

import asyncio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_in_repo(argument: str) -> Path:
    path = (REPO_ROOT / argument.strip()).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"ruta fuera del repositorio: {argument}")
    return path


class ReadFileTool:
    name = "read"
    description = "Lee un fichero del repositorio"

    async def run(self, argument: str) -> str:
        if not argument.strip():
            raise ValueError("uso: /read <ruta>")
        path = _resolve_in_repo(argument)
        if not path.is_file():
            raise FileNotFoundError(f"no existe (o no es un fichero): {argument}")
        return path.read_text(encoding="utf-8", errors="replace")


class RunPytestTool:
    name = "test"
    description = "Ejecuta pytest"

    async def run(self, argument: str) -> str:
        target = argument.strip() or "tests"
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", target, "-v",
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        lines = []
        async for raw_line in proc.stdout:
            lines.append(raw_line.decode(errors="replace").rstrip())
        await proc.wait()
        return "\n".join(lines)


class DiffTool:
    name = "diff"
    description = "Muestra el git diff de un fichero"

    async def run(self, argument: str) -> str:
        if not argument.strip():
            raise ValueError("uso: /diff <ruta>")
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--", argument.strip(),
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace") or "git diff failed")
        return stdout.decode(errors="replace")
