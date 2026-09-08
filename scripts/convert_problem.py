import argparse
import html as html_module
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional


def extract_limits(text: str) -> Optional[str]:
    match = re.search(
        r"実行時間制限\s*:\s*(.*?)\s*/\s*"
        r"メモリ制限\s*:\s*(.*?)\s*</p\s*>",
        text,
        flags=re.S | re.I,
    )
    if not match:
        return None

    values = []
    for value in match.groups():
        value = re.sub(r"<[^>]*>", "", value)
        value = html_module.unescape(value)
        values.append(re.sub(r"\s+", " ", value).strip())

    return f"実行時間制限: {values[0]} / メモリ制限: {values[1]}"


def insert_limits_after_title(md: str, limits: str) -> str:
    return re.sub(
        r"^(# [^\n]+\n)",
        lambda match: f"{match.group(1)}\n{limits}\n",
        md,
        count=1,
    )


class MDConverter(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out = []
        self.list_depth = 0
        self.in_pre = False
        self.pre_buf = []
        self.pre_lang = ""
        self.in_heading = None  # type: Optional[int]
        self.heading_buf = []
        self.in_a = False
        self.a_href = None  # type: Optional[str]
        self.a_buf = []
        self.in_table = False
        self.table_buf = []

    def _write(self, s: str) -> None:
        if not s:
            return
        if self.in_heading is not None:
            self.heading_buf.append(s)
        elif self.in_a:
            self.a_buf.append(s)
        elif self.in_pre:
            self.pre_buf.append(s)
        elif self.in_table:
            self.table_buf.append(s)
        else:
            self.out.append(s)

    def _ensure_blankline(self) -> None:
        if self.in_pre or self.in_table or self.in_heading is not None:
            return
        cur = "".join(self.out)
        if not cur.endswith("\n\n"):
            if cur.endswith("\n"):
                self.out.append("\n")
            else:
                self.out.append("\n\n")

    def handle_starttag(self, tag, attrs) -> None:
        attrs = dict(attrs)

        if self.in_table and tag != "table":
            self._write(self.get_starttag_text() or "")
            return

        if tag == "table":
            self._ensure_blankline()
            self.in_table = True
            self.table_buf = [self.get_starttag_text() or "<table>"]
            return

        if tag in ("div", "section"):
            return

        if tag == "h3":
            self._ensure_blankline()
            self.in_heading = 2
            self.heading_buf = []
            return

        if tag == "h4":
            self._ensure_blankline()
            self.in_heading = 3
            self.heading_buf = []
            return

        if tag == "p":
            self._ensure_blankline()
            return

        if tag == "ul":
            self._ensure_blankline()
            self.list_depth += 1
            return

        if tag == "li":
            indent = "  " * max(0, self.list_depth - 1)
            self._write(f"{indent}- ")
            return

        if tag == "br":
            self._write("\n")
            return

        if tag == "pre":
            self._ensure_blankline()
            self.in_pre = True
            self.pre_buf = []
            cls = attrs.get("class", "")
            self.pre_lang = "python" if "prettyprint" in cls else ""
            return

        if tag == "strong":
            self._write("**")
            return

        if tag == "em":
            self._write("*")
            return

        if tag == "var":
            if not self.in_pre:
                self._write("$")
            return

        if tag == "a":
            self.in_a = True
            self.a_href = attrs.get("href", "")
            self.a_buf = []
            return

        if tag == "img":
            src = attrs.get("src", "")
            if src:
                self._ensure_blankline()
                self._write(f"![]({src})\n")
            return

        if tag == "figure":
            self._ensure_blankline()
            return

        if tag == "details":
            self._ensure_blankline()
            self._write("<details>\n")
            return

        if tag == "summary":
            self._write("<summary>")
            return

    def handle_endtag(self, tag) -> None:
        if self.in_table and tag != "table":
            self._write(f"</{tag}>")
            return

        if tag == "table":
            self.table_buf.append(f"</{tag}>")
            self.out.append("".join(self.table_buf))
            self.in_table = False
            self.table_buf = []
            self._ensure_blankline()
            return

        if tag in ("h3", "h4"):
            lvl = self.in_heading or 2
            title = "".join(self.heading_buf)
            title = html_module.unescape(title)
            title = re.sub(r"\s+", " ", title).strip()
            self.out.append("#" * lvl + " " + title + "\n\n")
            self.in_heading = None
            self.heading_buf = []
            return

        if tag == "p":
            self._write("\n\n")
            return

        if tag == "ul":
            self.list_depth = max(0, self.list_depth - 1)
            self._write("\n")
            return

        if tag == "li":
            self._write("\n")
            return

        if tag == "pre":
            content = "".join(self.pre_buf)
            content = html_module.unescape(content)
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            content = content.strip("\n")
            fence = "```"
            lang = self.pre_lang
            self.out.append(f"{fence}{lang}\n{content}\n{fence}\n\n")
            self.in_pre = False
            self.pre_buf = []
            self.pre_lang = ""
            return

        if tag == "strong":
            self._write("**")
            return

        if tag == "em":
            self._write("*")
            return

        if tag == "var":
            if not self.in_pre:
                self._write("$")
            return

        if tag == "a":
            href = self.a_href or ""
            label = "".join(self.a_buf)
            label = html_module.unescape(label)
            label = re.sub(r"\s+", " ", label).strip()
            if not label:
                label = href
            self.out.append(f"[{label}]({href})")
            self.in_a = False
            self.a_href = None
            self.a_buf = []
            return

        if tag == "summary":
            self._write("</summary>\n")
            return

        if tag == "details":
            self._write("</details>\n\n")
            return

    def handle_data(self, data) -> None:
        self._write(data)

    def handle_entityref(self, name) -> None:
        self._write(f"&{name};")

    def handle_charref(self, name) -> None:
        self._write(f"&#{name};")


def main() -> None:
    parser = argparse.ArgumentParser(description="AtCoder の問題 HTML を Markdown に変換します。")
    parser.add_argument(
        "src",
        nargs="?",
        type=Path,
        default=Path("problem.html"),
        help="入力 HTML ファイル（既定値: problem.html）",
    )
    parser.add_argument(
        "dst",
        nargs="?",
        type=Path,
        default=Path("problem.md"),
        help="出力 Markdown ファイル（既定値: problem.md）",
    )
    args = parser.parse_args()
    src = args.src
    dst = args.dst

    text = src.read_text(encoding="utf-8", errors="replace")
    limits = extract_limits(text)

    m = re.search(
        r'<span class="lang-ja">(.*?)</span>\s*<span class="lang-en">',
        text,
        flags=re.S,
    )
    if not m:
        raise SystemExit("Could not find lang-ja block")

    ja_html = m.group(1)
    ja_html = re.sub(r"<script.*?</script>", "", ja_html, flags=re.S | re.I)
    ja_html = re.sub(r"<style.*?</style>", "", ja_html, flags=re.S | re.I)

    conv = MDConverter()
    conv.feed("<root>" + ja_html + "</root>")
    md = "".join(conv.out)

    md = html_module.unescape(md)

    # Normalize display math delimiters.
    md = md.replace("\\[", "$$\n").replace("\\]", "\n$$")

    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = md.strip() + "\n"

    if not md.startswith("# "):
        md = "# A - Multi-Player Territory Game\n\n" + md

    if limits is not None:
        md = insert_limits_after_title(md, limits)

    dst.write_text(md, encoding="utf-8")
    print(f"Wrote {dst} ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()