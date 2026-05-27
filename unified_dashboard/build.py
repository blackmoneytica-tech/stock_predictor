#!/usr/bin/env python3
"""
Unified Dashboard builder — merges the US (G5_22) and KR (V25) dashboards into a
single static index.html with tab switching, while keeping both backends untouched.

Sources (read-only):
  - US:  ../../g5-22-trader/pages/public/index.html   (inline <style> + body + inline <script>)
  - KR:  ../kr_dashboard/index.html  (+ style.css + script.js)   data is fetched cross-origin from kr-v25.pages.dev

Isolation strategy (see unified_dashboard/README.md):
  - CSS: every selector in each dashboard is scoped under #view-us / #view-kr  (no class collisions)
  - JS:  US stays global (it uses inline onclick=...); KR is wrapped in an IIFE (it uses addEventListener)
  - IDs: KR's 3 colliding ids (last-update, picks-table, picks-body) are renamed with a kr- prefix
  - KR data: fetch('data/...') is rewritten to fetch(KR_BASE + '/data/...') cross-origin

Run:  python build.py   ->  writes unified_dashboard/index.html
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
# US 원본은 g5-22-trader (별도 위치, repo 밖) — CI에서 접근 불가하므로 snapshot 병행.
# 로컬 빌드: 원본 읽고 us_source.html 갱신. CI 빌드(원본 없음): us_source.html 사용.
US_HTML_ORIG = os.path.normpath(os.path.join(HERE, "..", "..", "g5-22-trader", "pages", "public", "index.html"))
US_SNAPSHOT = os.path.join(HERE, "us_source.html")
KR_HTML = os.path.normpath(os.path.join(HERE, "..", "kr_dashboard", "index.html"))
KR_CSS  = os.path.normpath(os.path.join(HERE, "..", "kr_dashboard", "style.css"))
KR_JS   = os.path.normpath(os.path.join(HERE, "..", "kr_dashboard", "script.js"))
OUT     = os.path.join(HERE, "index.html")

KR_DATA_ORIGIN = "https://kr-v25.pages.dev"   # data SSOT — kept fresh by GitHub Actions


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------- CSS scoping
def prefix_selector(sel, root):
    s = sel.strip()
    low = s.lower()
    if low in ("body", "html", ":root"):
        return root
    if low.startswith("body "):
        return root + " " + s[5:]
    if low.startswith("html "):
        return root + " " + s[5:]
    return root + " " + s


def prefix_selector_list(sel, root):
    return ", ".join(prefix_selector(p, root) for p in sel.split(",") if p.strip())


def scope_css(css, root):
    """Prefix every top-level rule's selector with `root`. @media/@supports are
    recursed into; @keyframes/@font-face/@charset/@import are left global."""
    out = []
    i, n = 0, len(css)
    while i < n:
        ch = css[i]
        if ch in " \t\r\n":
            out.append(ch); i += 1; continue
        if css[i:i + 2] == "/*":
            end = css.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(css[i:end]); i = end; continue
        # read up to '{' or ';'
        j = i
        while j < n and css[j] not in "{;":
            if css[j:j + 2] == "/*":
                e = css.find("*/", j + 2); j = n if e == -1 else e + 2; continue
            j += 1
        if j >= n:
            out.append(css[i:]); break
        if css[j] == ";":          # @import / @charset ...
            out.append(css[i:j + 1]); i = j + 1; continue
        sel = css[i:j].strip()
        # find matching close brace
        depth, k = 1, j + 1
        while k < n and depth > 0:
            if css[k:k + 2] == "/*":
                e = css.find("*/", k + 2); k = n if e == -1 else e + 2; continue
            if css[k] == "{": depth += 1
            elif css[k] == "}": depth -= 1
            k += 1
        block = css[j + 1:k - 1]
        low = sel.lower()
        if low.startswith(("@keyframes", "@-webkit-keyframes", "@font-face", "@page", "@charset")):
            out.append(sel + " {" + block + "}")
        elif low.startswith(("@media", "@supports", "@container")):
            out.append(sel + " {" + scope_css(block, root) + "}")
        else:
            out.append(prefix_selector_list(sel, root) + " {" + block + "}")
        i = k
    return "".join(out)


# ---------------------------------------------------------------- extraction
def between(text, start, end, after=0):
    a = text.index(start, after) + len(start)
    b = text.index(end, a)
    return text[a:b], b


def extract_us(html):
    style, _ = between(html, "<style>", "</style>")
    body_start = html.index("<body>") + len("<body>")
    script_open = html.index("<script>", body_start)
    body = html[body_start:script_open]
    script, _ = between(html, "<script>", "</script>", after=script_open)
    return style, body, script


def extract_kr_body(html):
    body_start = html.index("<body>") + len("<body>")
    script_tag = html.index("<script", body_start)
    return html[body_start:script_tag]


# ---------------------------------------------------------------- KR fixups
def fix_kr_body(body):
    body = body.replace('id="last-update"', 'id="kr-last-update"')
    body = body.replace('id="picks-table"', 'id="kr-picks-table"')
    body = body.replace('id="picks-body"', 'id="kr-picks-body"')
    return body


def fix_kr_js(js):
    # rename colliding ids
    js = js.replace("getElementById('last-update')", "getElementById('kr-last-update')")
    js = js.replace("getElementById('picks-body')", "getElementById('kr-picks-body')")
    js = js.replace("getElementById('picks-table')", "getElementById('kr-picks-table')")
    # data fetch -> cross-origin
    js = js.replace("fetch('data/", "fetch(KR_BASE + '/data/")
    # wrap whole file in an IIFE so KR globals (saveCapital, SECTOR_NAMES, ...) don't leak
    header = "(function(){\n'use strict';\nconst KR_BASE = %r;\n" % KR_DATA_ORIGIN
    return header + js + "\n})();\n"


def fix_kr_css(css):
    # #order-table id selectors stay unique within #view-kr; just scope normally.
    return scope_css(css, "#view-kr")


# ---------------------------------------------------------------- assemble
def main():
    # US 소스: 로컬 원본 우선(있으면 snapshot 갱신), 없으면 snapshot(CI) 사용
    if os.path.exists(US_HTML_ORIG):
        us_raw = read(US_HTML_ORIG)
        with open(US_SNAPSHOT, "w", encoding="utf-8") as f:
            f.write(us_raw)
        print(f"  US: 원본 읽음 + snapshot 갱신 ({US_SNAPSHOT})")
    elif os.path.exists(US_SNAPSHOT):
        us_raw = read(US_SNAPSHOT)
        print(f"  US: snapshot 사용 (원본 없음 — CI 모드)")
    else:
        raise FileNotFoundError(f"US 소스 없음: {US_HTML_ORIG} 도 {US_SNAPSHOT} 도 없음")
    us_style, us_body, us_script = extract_us(us_raw)
    kr_body = fix_kr_body(extract_kr_body(read(KR_HTML)))
    kr_css = fix_kr_css(read(KR_CSS))
    kr_js = fix_kr_js(read(KR_JS))
    us_css = scope_css(us_style, "#view-us")

    # Planner (3rd tab) — own files, pre-scoped under #view-planner, injected raw
    planner_html = read(os.path.join(HERE, "planner.html"))
    planner_css  = read(os.path.join(HERE, "planner.css"))
    planner_js   = read(os.path.join(HERE, "planner.js"))

    shell_css = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #0a0e1a; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Apple SD Gothic Neo", sans-serif; }
  #tabbar {
    position: sticky; top: 0; z-index: 1000;
    display: flex; align-items: center; gap: 8px;
    padding: 10px 16px; background: #0d1320cc; backdrop-filter: blur(8px);
    border-bottom: 1px solid #1f2937;
  }
  #tabbar .brand { font-weight: 800; font-size: 15px; color: #e7eaf0; margin-right: auto; letter-spacing: -0.3px; }
  #tabbar button.tab {
    background: #1f2937; color: #cbd5e1; border: 1px solid #374151;
    padding: 8px 16px; border-radius: 999px; font-size: 14px; font-weight: 700; cursor: pointer;
  }
  #tabbar button.tab:hover { background: #283344; }
  #tabbar button.tab.active { background: #2563eb; color: #fff; border-color: #3b82f6; }
  #view-us, #view-kr, #view-planner { display: none; }
  body.show-us #view-us { display: block; }
  body.show-kr #view-kr { display: block; }
  body.show-planner #view-planner { display: block; }
  @media (max-width: 600px) {
    #tabbar { padding: 8px 10px; gap: 6px; }
    #tabbar .brand { font-size: 13px; }
    #tabbar button.tab { padding: 6px 12px; font-size: 13px; }
  }
"""

    shell_js = """
function showTab(which) {
  ['us','kr','planner'].forEach(function(w){
    document.body.classList.toggle('show-'+w, w === which);
    var b = document.getElementById('tab-'+w);
    if (b) b.classList.toggle('active', w === which);
  });
  try { localStorage.setItem('trader_hub_tab', which); } catch (e) {}
}
showTab((function(){ try { return localStorage.getItem('trader_hub_tab') || 'us'; } catch(e){ return 'us'; } })());
"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>📊 Trader Hub — US G5_22 + KR V25</title>
<style>
/* ===== shell ===== */
{shell_css}
/* ===== US (G5_22) — scoped under #view-us ===== */
{us_css}
/* ===== KR (V25) — scoped under #view-kr ===== */
{kr_css}
/* ===== Planner — pre-scoped under #view-planner ===== */
{planner_css}
</style>
</head>
<body class="show-us">

<nav id="tabbar">
  <span class="brand">📊 Trader Hub</span>
  <button id="tab-us" class="tab active" onclick="showTab('us')">🇺🇸 US · G5_22</button>
  <button id="tab-kr" class="tab" onclick="showTab('kr')">🇰🇷 KR · V25</button>
  <button id="tab-planner" class="tab" onclick="showTab('planner')">🧮 플래너</button>
</nav>

<div id="view-us">
{us_body}
</div>

<div id="view-kr">
{kr_body}
</div>

<div id="view-planner">
{planner_html}
</div>

<script>
/* ===== shell tab switching ===== */
{shell_js}
</script>
<script>
/* ===== US (G5_22) dashboard — global scope (uses inline onclick handlers) ===== */
{us_script}
</script>
<script>
/* ===== KR (V25) dashboard — IIFE isolated, data via cross-origin fetch ===== */
{kr_js}
</script>
<script>
/* ===== Position Planner — IIFE isolated, /api/quote + regime ===== */
{planner_js}
</script>
</body>
</html>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
