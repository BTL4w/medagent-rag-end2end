"""
YouMed content cleaner
Transforms content field in JSONL using custom rules.
"""

import json
import logging
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_addison_logic(text: str) -> str:
    # 0. Bỏ phần text trước \n\n<h2> đầu tiên
    marker = "\n\n<h2>"
    if marker in text:
        text = text.split(marker, 1)[1]
        text = marker + text

    # 1. Xử lý Header (Mở + Đóng)
    text = text.replace("\n\n<h2>", "\n\n## ")
    text = text.replace("\n\n<h3>", "\n\n### ")

    text = text.replace("</h2>\n\n", "\n")
    text = text.replace("</h3>\n\n", "\n")

    # 2. Xử lý List
    text = text.replace("\n\n<li>", "\n* ")
    text = text.replace("</li>", "")

    return text


def strip_remaining_html_tags(text: str) -> str:
    """Remove any leftover angle-bracket tags after markdown conversion."""
    return _HTML_TAG_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def strip_leading_toc_plain_text(text: str) -> str:
    """
    YouMed plain-text exports often start with:
    - optional duplicate title
    - 'Nội dung bài viết' / 'Toggle'
    - a block of short lines ending with '?' (TOC), then real paragraphs.

    HTML crawls use <h2> and should not match this path strongly.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and re.match(r"^\s*<[^>]+>", lines[i]):
        return text

    # 1) Optional duplicate headline (not TOC label), before boilerplate
    if i < len(lines) and lines[i].strip() not in ("Nội dung bài viết", "Toggle"):
        if not lines[i].strip().endswith("?"):
            i += 1

    # 2) Site UI lines
    while i < len(lines) and lines[i].strip() in ("Nội dung bài viết", "Toggle", ""):
        i += 1

    # 3) Khối mục lục: nhiều dòng ngắn (có hoặc không có '?') trước đoạn mở đầu dài
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if len(s) >= 120:
            break
        i += 1

    return "\n".join(lines[i:]).strip()


def clean_youmed_content(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""

    text = text.replace("\r\n", "\n")

    head = "\n".join(text.splitlines()[:12])
    if "Nội dung bài viết" in head or (
        "Toggle" in head and "<h2>" not in head and "<h3>" not in head
    ):
        text = strip_leading_toc_plain_text(text)

    if "<h2>" in text or "<h3>" in text or "<li>" in text:
        text = clean_addison_logic(text)

    text = strip_remaining_html_tags(text)
    text = normalize_whitespace(text)
    return text


class YouMedContentCleaner:
    """Clean content field from JSONL and write to processed output."""

    def __init__(
        self,
        input_file: str = "data/raw/youmed_articles_test.jsonl",
        output_dir: str = "data/processed",
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = (
            self.output_dir / f"{self.input_file.stem}_processed{self.input_file.suffix}"
        )

    def _iter_jsonl(self) -> Iterable[Dict[str, Any]]:
        with self.input_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def clean(self) -> None:
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        count = 0
        with self.output_file.open("w", encoding="utf-8") as out:
            for record in self._iter_jsonl():
                content = record.get("content", "")
                if isinstance(content, str) and content:
                    record["content"] = clean_youmed_content(content)
                json.dump(record, out, ensure_ascii=False)
                out.write("\n")
                count += 1

        logger.info(f"Cleaned {count} records to {self.output_file}")


if __name__ == "__main__":
    input_files = glob("data/raw/youmed_articles*.jsonl")
    for input_file in sorted(input_files):
        cleaner = YouMedContentCleaner(input_file=input_file)
        cleaner.clean()
