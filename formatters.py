# formatters.py
import html
import json
import re
from typing import List, Optional, Tuple

CODE_FENCE_RE = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)
MD_TABLE_RE = re.compile(
    r"(^\|.+\|\s*$\n^\|[-:\s|]+\|\s*$\n(?:^\|.*\|\s*$\n?)+)",
    re.MULTILINE,
)


def esc_html(s: str) -> str:
    return html.escape(s)




def parse_json_rows(text: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    """
    If the whole message is JSON list[dict], return (headers, rows), else None.
    """
    t = text.strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    try:
        obj = json.loads(t)
    except Exception:
        return None

    if not isinstance(obj, list) or not obj or not all(isinstance(x, dict) for x in obj):
        return None

    # stable union of keys
    keys: List[str] = []
    seen = set()
    for d in obj:
        for k in d.keys():
            ks = str(k)
            if ks not in seen:
                seen.add(ks)
                keys.append(ks)

    rows: List[List[str]] = []
    for d in obj:
        rows.append([str(d.get(k, "")) for k in keys])

    return keys, rows


def md_table_to_grid(md_block: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    """
    Convert a markdown table block to (headers, rows).
    """
    lines = [ln.strip() for ln in md_block.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    if not (lines[0].startswith("|") and "|" in lines[0]):
        return None
    if not lines[1].startswith("|"):
        return None

    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    body_lines = lines[2:]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in body_lines]
    return headers, rows


def to_rich_html(text: str) -> str:
    """
    For normal chat rendering: code blocks and markdown tables become HTML.
    NOTE: CSV/JSON are handled in UI via QTableWidget (sheet), not here.
    """
    out = text

    # Convert markdown tables to simple HTML table (for chat bubble, optional)
    def _table_repl(m):
        grid = md_table_to_grid(m.group(1))
        if not grid:
            return esc_html(m.group(1)).replace("\n", "<br/>")
        headers, rows = grid

        def cell(tag: str, val: str) -> str:
            return (
                f"<{tag} style='border:1px solid rgba(255,255,255,0.16);"
                f" padding:6px 10px; white-space:nowrap;'>"
                f"{esc_html(val)}"
                f"</{tag}>"
            )

        thead = "<tr>" + "".join(cell("th", h) for h in headers) + "</tr>"
        tbody = "".join("<tr>" + "".join(cell("td", v) for v in row) + "</tr>" for row in rows)

        return (
            "<div style='margin-top:8px; overflow:auto; max-width:100%;'>"
            "<table style='border-collapse:collapse; width:max-content; min-width:100%;"
            " background:rgba(255,255,255,0.03); border-radius:12px;'>"
            f"<thead style='background:rgba(255,255,255,0.06); position:sticky; top:0;'>{thead}</thead>"
            f"<tbody>{tbody}</tbody>"
            "</table></div>"
        )

    out = re.sub(MD_TABLE_RE, _table_repl, out)

    # Convert fenced code blocks
    def _code_repl(m):
        lang = (m.group(1) or "").strip()
        code = m.group(2)
        badge = (
            f"<div style='opacity:0.8; font-weight:800; margin-bottom:6px;'>{esc_html(lang)}</div>"
            if lang
            else ""
        )
        return (
            "<div style='margin-top:8px; margin-bottom:8px;'>"
            f"{badge}"
            "<pre style='padding:12px; border-radius:14px; background:rgba(0,0,0,0.35);"
            " border:1px solid rgba(255,255,255,0.12); overflow:auto;'>"
            f"<code>{esc_html(code)}</code></pre></div>"
        )

    out = re.sub(CODE_FENCE_RE, _code_repl, out)

    # Escape if we didn't inject HTML
    if out == text:
        return esc_html(text).replace("\n", "<br/>")

    return out.replace("\n", "<br/>")
