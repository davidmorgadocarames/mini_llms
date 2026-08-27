import pytest

from coconut_tui.tools.builtin import REPO_ROOT, DiffTool, ReadFileTool, RunPytestTool


@pytest.mark.asyncio
async def test_read_file_tool_reads_a_real_repo_file():
    tool = ReadFileTool()
    content = await tool.run("README.md")
    assert "Coconut" in content


@pytest.mark.asyncio
async def test_read_file_tool_rejects_path_outside_repo():
    tool = ReadFileTool()
    with pytest.raises(ValueError):
        await tool.run("../outside.txt")


@pytest.mark.asyncio
async def test_read_file_tool_raises_on_missing_file():
    tool = ReadFileTool()
    with pytest.raises(FileNotFoundError):
        await tool.run("this_file_does_not_exist.txt")


@pytest.mark.asyncio
async def test_diff_tool_reports_a_real_uncommitted_change():
    # Self-contained: mutates a tracked file, checks DiffTool sees the real
    # git diff, then restores the file — doesn't depend on ambient repo state.
    target = REPO_ROOT / ".gitignore"
    original = target.read_text(encoding="utf-8")
    try:
        target.write_text(original + "\n# coconut_tui diff-tool test marker\n", encoding="utf-8")
        diff = await DiffTool().run(".gitignore")
        assert "coconut_tui diff-tool test marker" in diff
    finally:
        target.write_text(original, encoding="utf-8")

    diff_after_revert = await DiffTool().run(".gitignore")
    assert diff_after_revert == ""


@pytest.mark.asyncio
async def test_run_pytest_tool_executes_real_pytest_and_reports_pass():
    tool = RunPytestTool()
    output = await tool.run("tests/test_depth_generator.py")
    assert "passed" in output.lower()
