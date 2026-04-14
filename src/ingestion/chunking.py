from __future__ import annotations

import json
import uuid
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter



@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    chunk_id: str
    enriched_content: str
    original_content: Optional[str] = None
    metadata: Optional[Dict] = None
    section: Optional[str] = None
    subsection: Optional[str] = None


class MarkdownChunker:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        headers_to_split: Optional[List[tuple]] = None,
    ) -> None:
        self.headers_to_split = headers_to_split or [("##", "section"), ("###", "subsection")]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._init_splitters()

    def _init_splitters(self) -> None:
        self._header_splitter = None
        self._recursive_splitter = None
        if MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter:
            self._header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split
            )
            self._recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def _fix_encoding(self, text: str) -> str:
        """Best-effort fix for common mojibake text."""
        if not text:
            return text
        suspicious_tokens = ("Ã", "Â", "â€", "â€™", "â€œ", "â€”", "�")
        if any(token in text for token in suspicious_tokens):
            try:
                repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
                if repaired and repaired.count("�") <= text.count("�"):
                    return repaired
            except Exception:
                pass
        return text

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and merge broken lines inside paragraphs."""
        text = self._fix_encoding(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        paragraphs: List[str] = []
        for block in re.split(r"\n\s*\n+", text):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            merged = " ".join(lines)
            merged = re.sub(r"\s+", " ", merged).strip()
            if merged:
                paragraphs.append(merged)
        return "\n\n".join(paragraphs).strip()

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r"(?<=[\.\!\?\:\;…])\s+", text.strip())
        return [part.strip() for part in parts if part and part.strip()]

    def _last_n_sentences(self, text: str, n: int = 2) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return ""
        return " ".join(sentences[-n:])

    def _count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text or ""))

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph by sentence, never cutting mid-sentence."""
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        sentences = self._split_sentences(paragraph)
        if not sentences:
            return [paragraph]
        parts: List[str] = []
        current = ""
        for sent in sentences:
            if not current:
                current = sent
                continue
            candidate = f"{current} {sent}"
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                parts.append(current.strip())
                current = sent
        if current.strip():
            parts.append(current.strip())
        return parts

    def _semantic_split(self, text: str) -> List[str]:
        """
        Split by paragraph/câu:
        - KHÔNG cắt giữa paragraph nếu paragraph <= chunk_size
        - KHÔNG cắt giữa câu
        - CHỈ tách khi > chunk_size
        - overlap: 2 câu cuối, chỉ khi đang cắt dở cùng 1 paragraph
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        units: List[Dict[str, object]] = []
        for paragraph_index, paragraph in enumerate(normalized.split("\n\n")):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            paragraph_parts = self._split_long_paragraph(paragraph)
            fragmented = len(paragraph_parts) > 1
            for part in paragraph_parts:
                units.append(
                    {
                        "text": part,
                        "paragraph_index": paragraph_index,
                        "fragmented": fragmented,
                    }
                )

        chunks: List[str] = []
        current = ""
        current_last_meta: Optional[Dict[str, object]] = None

        for unit in units:
            unit_text = str(unit["text"])
            if not current:
                candidate = unit_text
            else:
                candidate = f"{current}\n\n{unit_text}"

            if current and len(candidate) > self.chunk_size:
                chunks.append(current.strip())
                should_overlap = False
                if current_last_meta is not None:
                    should_overlap = bool(
                        current_last_meta.get("fragmented")
                        and unit.get("fragmented")
                        and current_last_meta.get("paragraph_index") == unit.get("paragraph_index")
                    )
                overlap_tail = self._last_n_sentences(current, n=2) if should_overlap else ""
                current = f"{overlap_tail} {unit_text}".strip() if overlap_tail else unit_text
            else:
                current = candidate
            current_last_meta = unit

        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _resolve_doc_id(self, metadata: Dict, supplied_doc_id: Optional[str]) -> str:
        # Chuẩn hóa theo yêu cầu: doc_id luôn dùng UUID.
        if supplied_doc_id:
            try:
                return str(uuid.UUID(str(supplied_doc_id)))
            except ValueError:
                pass
        return str(uuid.uuid4())

    def _merge_short_chunks(self, texts: List[str], min_tokens: int = 150) -> List[str]:
        """Merge tiny chunks with previous/next neighbor within same segment."""
        if not texts:
            return []

        merged = [text.strip() for text in texts if text and text.strip()]
        i = 0
        while i < len(merged):
            if self._count_tokens(merged[i]) >= min_tokens:
                i += 1
                continue
            if len(merged) == 1:
                break

            if i > 0:
                merged[i - 1] = f"{merged[i - 1]}\n\n{merged[i]}".strip()
                merged.pop(i)
                i = max(i - 1, 0)
                continue

            merged[i + 1] = f"{merged[i]}\n\n{merged[i + 1]}".strip()
            merged.pop(i)
        return merged

    def _split_by_headers_fallback(self, content: str) -> List[Dict[str, Optional[str]]]:
        lines = content.splitlines()
        section = None
        subsection = None
        buffer: List[str] = []
        segments: List[Dict[str, Optional[str]]] = []

        def flush() -> None:
            if buffer:
                segments.append(
                    {
                        "section": section,
                        "subsection": subsection,
                        "text": "\n".join(buffer).strip(),
                    }
                )
                buffer.clear()

        for line in lines:
            if line.startswith("### "):
                flush()
                subsection = line.replace("###", "", 1).strip() or None
                continue
            if line.startswith("## "):
                flush()
                section = line.replace("##", "", 1).strip() or None
                subsection = None
                continue
            buffer.append(line)

        flush()
        return [seg for seg in segments if seg.get("text")]

    def _split_text_fallback(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(text_length, start + self.chunk_size)
            chunks.append(text[start:end])
            if end == text_length:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _enrich_text(
        self,
        chunk_text: str,
        metadata: Dict,
        section: Optional[str],
        subsection: Optional[str],
    ) -> str:
        keyword = (metadata or {}).get("keyword") or (metadata or {}).get("title") or "Không rõ"
        if section and subsection:
            section_label = f"{section} ({subsection})"
        elif section:
            section_label = section
        elif subsection:
            section_label = subsection
        else:
            section_label = "Không rõ"
        return f"Tiêu đề: {keyword}\nMục: {section_label}\nNội dung:\n{chunk_text.strip()}"

    def chunk_document(self, content: str, metadata: Dict, doc_id: Optional[str] = None) -> List[Chunk]:
        if not content:
            return []

        chunks: List[Chunk] = []
        resolved_doc_id = self._resolve_doc_id(metadata, doc_id)
        chunk_index = 0

        if self._header_splitter and self._recursive_splitter:
            header_docs = self._header_splitter.split_text(content)
            for doc in header_docs:
                section = doc.metadata.get("section")
                subsection = doc.metadata.get("subsection")
                chunk_texts = self._merge_short_chunks(self._semantic_split(doc.page_content), min_tokens=150)
                for sub_chunk in chunk_texts:
                    chunk_id = f"{resolved_doc_id}_{chunk_index}"
                    enriched = self._enrich_text(sub_chunk, metadata, section, subsection)
                    chunks.append(
                        Chunk(
                            doc_id=resolved_doc_id,
                            chunk_index=chunk_index,
                            chunk_id=chunk_id,
                            enriched_content=enriched,
                            original_content=sub_chunk,
                            metadata=metadata,
                            section=section,
                            subsection=subsection,
                        )
                    )
                    chunk_index += 1
            return chunks

        for segment in self._split_by_headers_fallback(content):
            chunk_texts = self._merge_short_chunks(self._semantic_split(segment["text"]), min_tokens=150)
            for sub_chunk in chunk_texts:
                chunk_id = f"{resolved_doc_id}_{chunk_index}"
                enriched = self._enrich_text(
                    sub_chunk,
                    metadata,
                    segment.get("section"),
                    segment.get("subsection"),
                )
                chunks.append(
                    Chunk(
                        doc_id=resolved_doc_id,
                        chunk_index=chunk_index,
                        chunk_id=chunk_id,
                        enriched_content=enriched,
                        original_content=sub_chunk,
                        metadata=metadata,
                        section=segment.get("section"),
                        subsection=segment.get("subsection"),
                    )
                )
                chunk_index += 1
        return chunks

    def process_jsonl(self, input_path: str) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                metadata = doc.get("metadata", {}) or {}
                doc_id = doc.get("doc_id") or doc.get("id") or metadata.get("doc_id")
                chunks = self.chunk_document(doc.get("content", ""), metadata, doc_id=doc_id)
                all_chunks.extend(chunks)
        return all_chunks
