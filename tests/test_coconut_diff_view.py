from coconut_tui.widgets.diff_view import render_diff

SAMPLE_DIFF = """diff --git a/x.py b/x.py
index 111..222 100644
--- a/x.py
+++ b/x.py
@@ -1,2 +1,2 @@
-old line
+new line
 unchanged"""


def test_render_diff_preserves_all_text():
    rendered = render_diff(SAMPLE_DIFF)
    assert rendered.plain == SAMPLE_DIFF


def test_render_diff_colors_additions_and_deletions():
    rendered = render_diff(SAMPLE_DIFF)
    styles_by_line = {}
    for line, spans in zip(SAMPLE_DIFF.splitlines(), rendered.split("\n")):
        styles_by_line[line] = str(spans.spans[0].style) if spans.spans else None

    assert styles_by_line["+new line"] == "green"
    assert styles_by_line["-old line"] == "red"
    assert styles_by_line["@@ -1,2 +1,2 @@"] == "cyan"
    assert styles_by_line[" unchanged"] is None


def test_render_diff_handles_empty_string():
    rendered = render_diff("")
    assert rendered.plain == ""
