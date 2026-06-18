from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


_TEXT_EXTRACT_LIMIT = 12000
_TEXT_FILE_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", ".yaml", ".yml"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def coerce_message_content_to_text(content: Any) -> str:
    """Convert structured chat content into plain text for text-only models.

    Some frontends send lists of content blocks like:
    [{"type": "text", "text": "..."}, {"type": "file", ...}]
    Local Ollama chat models do not support those block types directly, so we
    flatten text blocks and replace non-text attachments with short markers.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, (int, float, bool)):
        return str(content)

    if isinstance(content, dict):
        return _coerce_block_to_text(content)

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = coerce_message_content_to_text(block).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    return str(content)


def extract_text_from_path(path: Path) -> str:
    """Extract text from a local file path using the same OCR-aware logic."""
    return _extract_text_from_path(path)


def sanitize_history_pairs(history: list[tuple[str, Any]]) -> list[tuple[str, str]]:
    """Return a text-only history list safe for text-only chat models."""
    return [(role, coerce_message_content_to_text(content)) for role, content in history]


def _coerce_block_to_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", "") or "").strip().lower()

    if not block_type:
        for key in ("text", "content", "value"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(block)

    if block_type == "text":
        text = block.get("text")
        if isinstance(text, str):
            return text
        return str(text or "")

    if block_type == "file":
        name = (
            block.get("filename")
            or block.get("file_name")
            or block.get("name")
            or "uploaded file"
        )
        extracted_text = _extract_text_from_file_block(block)
        if extracted_text:
            return f"[Attached file: {name}]\n{extracted_text}"
        return f"[Attached file: {name}]"

    if block_type in {"image", "image_url", "input_image"}:
        return "[Attached image]"

    if block_type in {"audio", "input_audio"}:
        return "[Attached audio]"

    if block_type in {"video", "input_video"}:
        return "[Attached video]"

    for key in ("text", "content", "caption", "name", "filename"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return f"[Unsupported content block: {block_type}]"


def _extract_text_from_file_block(block: dict[str, Any]) -> str:
    """Best-effort text extraction from a structured file block.

    Supported sources:
    - Local path/URI fields (path, file_path, uri, etc.)
    - Inline base64 payloads in common fields (data, content, base64)
    """
    path = _extract_local_file_path(block)
    if path and path.exists() and path.is_file():
        return _extract_text_from_path(path)

    inline_bytes, suffix = _extract_inline_file_bytes(block)
    if inline_bytes:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(inline_bytes)
                tmp_path = Path(temp_file.name)
            return _extract_text_from_path(tmp_path)
        except Exception:
            return ""
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
    return ""


def _extract_local_file_path(block: dict[str, Any]) -> Path | None:
    candidates: list[Any] = [
        block.get("path"),
        block.get("file_path"),
        block.get("filepath"),
        block.get("uri"),
        block.get("url"),
    ]

    source = block.get("source")
    if isinstance(source, dict):
        candidates.extend(
            [
                source.get("path"),
                source.get("file_path"),
                source.get("filepath"),
                source.get("uri"),
                source.get("url"),
            ]
        )

    for candidate in candidates:
        path = _candidate_to_path(candidate)
        if path is not None:
            return path
    return None


def _candidate_to_path(candidate: Any) -> Path | None:
    if not isinstance(candidate, str):
        return None
    raw = candidate.strip()
    if not raw:
        return None

    if raw.lower().startswith("file://"):
        try:
            parsed = urlparse(raw)
            path_str = unquote(parsed.path or "")
            if os.name == "nt" and path_str.startswith("/"):
                path_str = path_str.lstrip("/")
            if path_str:
                return Path(path_str)
        except Exception:
            return None

    if "\n" in raw or "\r" in raw:
        return None

    return Path(raw)


def _extract_inline_file_bytes(block: dict[str, Any]) -> tuple[bytes | None, str]:
    filename = str(
        block.get("filename")
        or block.get("file_name")
        or block.get("name")
        or "uploaded"
    )
    suffix = Path(filename).suffix or ".bin"

    sources: list[Any] = [block.get("data"), block.get("content"), block.get("base64")]
    nested_source = block.get("source")
    if isinstance(nested_source, dict):
        sources.extend([nested_source.get("data"), nested_source.get("content"), nested_source.get("base64")])

    for raw in sources:
        data = _decode_bytes_payload(raw)
        if data:
            return data, suffix
    return None, suffix


def _decode_bytes_payload(raw: Any) -> bytes | None:
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        if value.startswith("data:") and "," in value:
            value = value.split(",", 1)[1]
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return None
    return None


def _extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in _TEXT_FILE_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception:
                return ""
        return _normalize_extracted_text(text)

    if suffix == ".pdf":
        return _extract_text_from_pdf(path)

    if suffix in _IMAGE_SUFFIXES:
        try:
            image_bytes = path.read_bytes()
        except Exception:
            return ""
        return _normalize_extracted_text(_ocr_image_bytes(image_bytes))

    return ""


def _extract_text_from_pdf(path: Path) -> str:
    try:
        import fitz
    except Exception:
        return ""

    pages_text: list[str] = []
    needs_ocr = True

    try:
        with fitz.open(str(path)) as pdf:
            for page in pdf:
                text = (page.get_text("text") or "").strip()
                if text:
                    pages_text.append(text)
                    needs_ocr = False

            if needs_ocr:
                ocr_chunks: list[str] = []
                for page in pdf:
                    try:
                        pix = page.get_pixmap(dpi=200, alpha=False)
                        ocr = _ocr_image_bytes(pix.tobytes("png"))
                        if ocr.strip():
                            ocr_chunks.append(ocr.strip())
                    except Exception:
                        continue
                if ocr_chunks:
                    pages_text.extend(ocr_chunks)
    except Exception:
        return ""

    return _normalize_extracted_text("\n\n".join(pages_text))


def _ocr_image_bytes(image_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""

    configured_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if configured_cmd:
        try:
            pytesseract.pytesseract.tesseract_cmd = configured_cmd
        except Exception:
            pass

    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
    except Exception:
        return ""
    return text.strip()


def _normalize_extracted_text(text: str) -> str:
    cleaned = "\n".join(line.rstrip() for line in str(text or "").splitlines())
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    if len(cleaned) > _TEXT_EXTRACT_LIMIT:
        return cleaned[:_TEXT_EXTRACT_LIMIT].rstrip() + "\n\n[...truncated extracted text...]"
    return cleaned
