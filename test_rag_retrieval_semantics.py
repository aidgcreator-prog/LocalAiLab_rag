from pathlib import Path
from types import SimpleNamespace
import io
import types

import agent
from ragsub_agent import tools as rag_tools
from ragsub_agent.prompts import RAG_SUB_INSTRUCTIONS
from PIL import Image


class _RecordingCollection:
    def __init__(self):
        self.records: dict[str, tuple[str, dict, str]] = {}
        self.deleted_ids: list[str] = []

    def count(self):
        return len(self.records)

    def peek(self, _count):
        ids = list(self.records.keys())
        docs = [self.records[item][0] for item in ids]
        metas = [self.records[item][1] for item in ids]
        return {"ids": ids, "documents": docs, "metadatas": metas}

    def delete(self, ids):
        for item in ids:
            if item in self.records:
                self.deleted_ids.append(item)
                self.records.pop(item, None)

    def upsert(self, documents, metadatas, ids):
        for document, metadata, doc_id in zip(documents, metadatas, ids):
            self.records[doc_id] = (document, metadata, doc_id)


class _PassthroughSplitter:
    def split_documents(self, docs):
        return docs


class _FakeEncoder:
    def rank(self, _query, texts, top_k):
        scored = []
        for index, text in enumerate(texts):
            score = 0.95 if "strong" in text else 0.15
            scored.append({"corpus_id": index, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def test_ingest_rag_paths_keeps_same_file_across_themes(monkeypatch, tmp_path):
    collection = _RecordingCollection()
    source_path = tmp_path / "shared.txt"
    source_path.write_text("alpha content", encoding="utf-8")

    monkeypatch.setattr(rag_tools, "_get_vector_collection", lambda project="Default": collection)
    monkeypatch.setattr(
        rag_tools,
        "_load_documents",
        lambda _path: ([SimpleNamespace(page_content="alpha content", metadata={})], {}),
    )
    monkeypatch.setattr(
        rag_tools,
        "_build_splitter",
        lambda **_kwargs: (_PassthroughSplitter(), "recursive"),
    )

    first = rag_tools.ingest_rag_paths([source_path], project="Proj", theme="Theme A")
    second = rag_tools.ingest_rag_paths([source_path], project="Proj", theme="Theme B")

    stored_themes = {meta[1]["theme"] for meta in collection.records.values()}

    assert first["loaded_files"] == 1
    assert second["loaded_files"] == 1
    assert stored_themes == {"Theme A", "Theme B"}
    assert len(collection.records) == 2


def test_ingest_rag_paths_replaces_only_stale_chunks_for_same_theme(monkeypatch, tmp_path):
    collection = _RecordingCollection()
    source_path = tmp_path / "shared.txt"
    source_path.write_text("version one", encoding="utf-8")
    payloads = iter(["version one", "version two"])

    monkeypatch.setattr(rag_tools, "_get_vector_collection", lambda project="Default": collection)
    monkeypatch.setattr(
        rag_tools,
        "_load_documents",
        lambda _path: ([SimpleNamespace(page_content=next(payloads), metadata={})], {}),
    )
    monkeypatch.setattr(
        rag_tools,
        "_build_splitter",
        lambda **_kwargs: (_PassthroughSplitter(), "recursive"),
    )

    rag_tools.ingest_rag_paths([source_path], project="Proj", theme="Theme A")
    result = rag_tools.ingest_rag_paths([source_path], project="Proj", theme="Theme A")

    assert result["stale_chunks_deleted"] == 1
    assert len(collection.deleted_ids) == 1
    assert len(collection.records) == 1
    stored_doc = next(iter(collection.records.values()))[0]
    assert "version two" in stored_doc


def test_ingest_web_search_results_stores_web_metadata(monkeypatch):
    collection = _RecordingCollection()

    monkeypatch.setattr(rag_tools, "_get_vector_collection", lambda project="Default": collection)
    monkeypatch.setattr(
        rag_tools,
        "_build_splitter",
        lambda **_kwargs: (_PassthroughSplitter(), "recursive"),
    )

    fake_research_tools = types.SimpleNamespace(
        TAVILY_AVAILABLE=True,
        tavily_client=types.SimpleNamespace(
            search=lambda query, max_results, topic: {
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "content": "Short snippet",
                        "published_date": "2026-04-26",
                    }
                ]
            }
        ),
        fetch_webpage_content=lambda url: "Full fetched web content about the topic.",
    )

    import sys
    sys.modules["research_agent.tools"] = fake_research_tools

    output = rag_tools.ingest_web_search_results.invoke(
        {
            "query": "example topic",
            "project": "WebProj",
            "theme": "News",
            "max_results": 1,
            "chunking_method": "recursive",
        }
    )

    assert "[OK] Web search ingestion complete" in output
    assert len(collection.records) == 1
    stored_doc, stored_meta, _stored_id = next(iter(collection.records.values()))
    assert "Full fetched web content about the topic." in stored_doc
    assert stored_meta["source_type"] == "web_search"
    assert stored_meta["web_url"] == "https://example.com/article"
    assert stored_meta["web_query"] == "example topic"


def test_rerank_chunks_selects_files_by_relevance(monkeypatch):
    monkeypatch.setattr(rag_tools, "_get_cross_encoder", lambda: _FakeEncoder())

    documents = ["weak chunk 1", "weak chunk 2", "strong signal"]
    metadatas = [
        {"source": "large.txt"},
        {"source": "large.txt"},
        {"source": "precise.txt"},
    ]
    ids = ["a", "b", "c"]

    ranked, diagnostics = rag_tools._rerank_chunks(
        query="example",
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        top_k=2,
        mode="Top-K Per File",
        max_files=1,
        min_score=0.0,
        return_details=True,
    )

    assert diagnostics["selected_files"] == ["precise.txt"]
    assert ranked == [("strong signal", {"source": "precise.txt"}, "c")]


def test_rerank_chunks_boosts_image_chunks_for_visual_queries(monkeypatch):
    monkeypatch.setattr(rag_tools, "_get_cross_encoder", lambda: _FakeEncoder())

    documents = ["strong signal text", "strong signal image"]
    metadatas = [
        {"source": "notes.txt", "modality": "text"},
        {"source": "diagram.pdf", "modality": "image"},
    ]
    ids = ["text-1", "img-1"]

    ranked, _diagnostics = rag_tools._rerank_chunks(
        query="Which figure or diagram shows the system flow?",
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        top_k=1,
        mode="Top-K Globally",
        max_files=2,
        min_score=0.0,
        return_details=True,
    )

    assert ranked == [("strong signal image", {"source": "diagram.pdf", "modality": "image"}, "img-1")]


def test_get_multimodal_ocr_status_reports_missing_binary(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "pytesseract":
            return SimpleNamespace()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(rag_tools.shutil, "which", lambda _name: None)
    monkeypatch.setenv("TESSERACT_CMD", "")

    status = rag_tools.get_multimodal_ocr_status()

    assert status["enabled"] is False
    assert status["state"] == "missing_binary"
    assert "TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe" in status["message"]
    assert status["remediation"].startswith("Install Tesseract")


def test_extract_image_ocr_text_returns_empty_when_tesseract_unavailable(monkeypatch):
    monkeypatch.setattr(rag_tools, "_tesseract_available", lambda: False)

    assert rag_tools._extract_image_ocr_text(b"fake-image-bytes") == ""


def test_get_multimodal_vision_status_reports_disabled_without_model(monkeypatch):
    monkeypatch.delenv("RAG_VISION_MODEL", raising=False)

    status = rag_tools.get_multimodal_vision_status()

    assert status["enabled"] is False
    assert status["state"] == "disabled"


def test_build_visual_summary_includes_vision_caption():
    summary, topics = rag_tools._build_visual_summary(
        b"fake-image",
        source="sample.pdf",
        page_number="2",
        image_index=1,
        page_text="Page discusses a system architecture diagram for onboarding.",
        ocr_text="",
        vision_caption="Architecture diagram showing API gateway, worker queue, and database.",
    )

    assert "vision caption: Architecture diagram showing API gateway, worker queue, and database." in summary
    assert isinstance(topics, str)


def test_build_visual_summary_includes_structured_visual_notes():
    summary, topics = rag_tools._build_visual_summary(
        b"fake-image",
        source="organization chart.pdf",
        page_number="1",
        image_index=1,
        page_text="General Directorate of Agriculture organization chart.",
        ocr_text="",
        vision_caption="Organization chart with photos and Khmer labels.",
        structured_notes="Top role shown: Director General. Top visible name is unreadable. Khmer script visible throughout.",
    )

    assert "structured visual notes: Top role shown: Director General." in summary
    assert isinstance(topics, str)


def test_should_extract_structured_visual_notes_matches_org_chart_markers():
    assert rag_tools._should_extract_structured_visual_notes(
        "General Directorate of Agriculture organization chart",
        "",
        "staff directory with hierarchy",
        "organization-chart.pdf",
    ) is True
    assert rag_tools._should_extract_structured_visual_notes(
        "architecture diagram",
        "",
        "system diagram",
        "diagram.png",
    ) is False


def test_extract_image_vision_caption_returns_empty_without_model(monkeypatch):
    monkeypatch.delenv("RAG_VISION_MODEL", raising=False)

    assert rag_tools._extract_image_vision_caption(b"fake-image-bytes") == ""


def test_extract_pdf_image_records_stores_vision_caption(monkeypatch, tmp_path):
    class _FakePdfPage:
        def get_images(self, full=True):
            return [(7,)]

    class _FakePdf:
        def __len__(self):
            return 1

        def load_page(self, index):
            assert index == 0
            return _FakePdfPage()

        def extract_image(self, xref):
            assert xref == 7
            image = Image.new("RGB", (200, 140), color="white")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return {"image": buffer.getvalue(), "ext": "png"}

        def close(self):
            return None

    import fitz

    monkeypatch.setattr(fitz, "open", lambda _path: _FakePdf())
    monkeypatch.setattr(rag_tools, "_extract_image_vision_caption", lambda _bytes: "System diagram with three services.")
    monkeypatch.setattr(rag_tools, "_extract_image_ocr_text", lambda _bytes: "")
    monkeypatch.setattr(rag_tools, "_should_extract_structured_visual_notes", lambda *args, **kwargs: False)

    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4 fake")
    page = SimpleNamespace(page_content="This page contains a deployment diagram.", metadata={"page_number": "1"})

    docs_payload, metas_payload, ids_payload = rag_tools._extract_pdf_image_records(
        file_path=source_path,
        pages=[page],
        project_name="Proj",
        theme_name="Theme",
        source_fingerprint="imgfingerprint",
        date_added="2026-04-26",
        doc_meta={},
    )

    assert len(docs_payload) == 1
    assert len(metas_payload) == 1
    assert len(ids_payload) == 1
    assert "vision caption: System diagram with three services." in docs_payload[0]
    assert metas_payload[0]["vision_caption"] == "System diagram with three services."
    assert metas_payload[0]["vision_caption_source"] == "ollama"


def test_assess_image_asset_quality_rejects_tiny_icon():
    import io

    image = Image.new("RGB", (24, 24), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    quality = rag_tools._assess_image_asset_quality(buffer.getvalue())

    assert quality["keep"] is False
    assert quality["reason"] == "too-small"


def test_extract_pdf_image_records_skips_low_value_images(monkeypatch, tmp_path):
    class _FakePdfPage:
        def get_images(self, full=True):
            return [(3,)]

    class _FakePdf:
        def __len__(self):
            return 1

        def load_page(self, index):
            assert index == 0
            return _FakePdfPage()

        def extract_image(self, xref):
            assert xref == 3
            image = Image.new("RGB", (24, 24), color="white")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return {"image": buffer.getvalue(), "ext": "png"}

        def close(self):
            return None

    import fitz

    monkeypatch.setattr(fitz, "open", lambda _path: _FakePdf())
    monkeypatch.setattr(rag_tools, "_extract_image_vision_caption", lambda _bytes: "tiny icon")
    monkeypatch.setattr(rag_tools, "_extract_image_ocr_text", lambda _bytes: "ignored")

    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4 fake")
    page = SimpleNamespace(page_content="This page has a small decorative icon.", metadata={"page_number": "1"})

    docs_payload, metas_payload, ids_payload = rag_tools._extract_pdf_image_records(
        file_path=source_path,
        pages=[page],
        project_name="Proj",
        theme_name="Theme",
        source_fingerprint="iconfilter",
        date_added="2026-04-26",
        doc_meta={},
    )

    assert docs_payload == []
    assert metas_payload == []
    assert ids_payload == []


def test_load_documents_audio_uses_transcription(monkeypatch, tmp_path):
    source_path = tmp_path / "clip.wav"
    source_path.write_bytes(b"RIFFfake")
    monkeypatch.setattr(rag_tools, "_transcribe_media_file", lambda _path: "transcribed speech")

    docs, meta = rag_tools._load_documents(source_path)

    assert len(docs) == 1
    assert docs[0].page_content == "transcribed speech"
    assert docs[0].metadata["media_kind"] == "audio"
    assert meta == {}


def test_load_documents_image_uses_image_placeholder(tmp_path):
    source_path = tmp_path / "chart.png"
    image = Image.new("RGB", (200, 140), color="white")
    image.save(source_path)

    docs, meta = rag_tools._load_documents(source_path)

    assert len(docs) == 1
    assert docs[0].metadata["media_kind"] == "image"
    assert "Standalone image file: chart.png" in docs[0].page_content
    assert meta == {}


def test_transcribe_media_file_falls_back_to_local_without_openai_key(monkeypatch, tmp_path):
    source_path = tmp_path / "clip.wav"
    source_path.write_bytes(b"RIFFfake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(rag_tools, "_transcribe_media_file_local", lambda path: f"local:{path.name}")

    transcript = rag_tools._transcribe_media_file(source_path)

    assert transcript == "local:clip.wav"


def test_extract_video_frame_records_creates_image_chunks(monkeypatch, tmp_path):
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"fake-video")

    monkeypatch.setattr(rag_tools, "_get_ffmpeg_executable", lambda: "ffmpeg")

    def _fake_run(command, capture_output, text):
        output_pattern = Path(command[-1])
        output_pattern.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (200, 140), color="white")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        (output_pattern.parent / "frame_0001.jpg").write_bytes(buffer.getvalue())
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(rag_tools.subprocess, "run", _fake_run)
    monkeypatch.setattr(rag_tools, "_extract_image_vision_caption", lambda _bytes: "Slide showing architecture flow.")
    monkeypatch.setattr(rag_tools, "_extract_image_ocr_text", lambda _bytes: "")
    monkeypatch.setattr(rag_tools, "_should_extract_structured_visual_notes", lambda *args, **kwargs: False)

    docs_payload, metas_payload, ids_payload = rag_tools._extract_video_frame_records(
        source_path,
        project_name="Proj",
        theme_name="Theme",
        source_fingerprint="vidfingerprint",
        date_added="2026-04-26",
        doc_meta={},
    )

    assert len(docs_payload) == 1
    assert len(metas_payload) == 1
    assert len(ids_payload) == 1
    assert "vision caption: Slide showing architecture flow." in docs_payload[0]
    assert metas_payload[0]["chunking_method"] == "multimodal_video_frame"
    assert metas_payload[0]["asset_path"].endswith("frame_0001.jpg")


def test_extract_standalone_image_records_creates_image_chunk(monkeypatch, tmp_path):
    source_path = tmp_path / "diagram.png"
    image = Image.new("RGB", (220, 150), color="white")
    image.save(source_path)

    monkeypatch.setattr(rag_tools, "_extract_image_vision_caption", lambda _bytes: "System diagram with three connected boxes.")
    monkeypatch.setattr(rag_tools, "_extract_image_ocr_text", lambda _bytes: "")
    monkeypatch.setattr(rag_tools, "_should_extract_structured_visual_notes", lambda *args, **kwargs: False)

    docs_payload, metas_payload, ids_payload = rag_tools._extract_standalone_image_records(
        source_path,
        project_name="Proj",
        theme_name="Theme",
        source_fingerprint="imgstandalone",
        date_added="2026-04-26",
        doc_meta={},
    )

    assert len(docs_payload) == 1
    assert len(metas_payload) == 1
    assert len(ids_payload) == 1
    assert "vision caption: System diagram with three connected boxes." in docs_payload[0]
    assert metas_payload[0]["chunking_method"] == "multimodal_image_file"
    assert metas_payload[0]["asset_path"].endswith("diagram.png")


def test_dedupe_multimodal_payloads_skips_duplicate_table_text():
    base_docs = ["Revenue table section sales q1 q2 q3 profit margin expenses"]
    base_metas = [{"source_fingerprint": "src1", "page_number": "2", "source": "sample.pdf"}]
    candidate_docs = ["Revenue table section sales q1 q2 q3 profit margin expenses"]
    candidate_metas = [{
        "source_fingerprint": "src1",
        "page_number": "2",
        "source": "sample.pdf",
        "modality": "table",
    }]
    candidate_ids = ["table-1"]

    docs, metas, ids = rag_tools._dedupe_multimodal_payloads(
        base_docs,
        base_metas,
        candidate_docs,
        candidate_metas,
        candidate_ids,
    )

    assert docs == []
    assert metas == []
    assert ids == []


def test_dedupe_multimodal_payloads_keeps_distinct_image_summary():
    base_docs = ["The page explains deployment requirements and onboarding steps."]
    base_metas = [{"source_fingerprint": "src1", "page_number": "3", "source": "sample.pdf"}]
    candidate_docs = ["Extracted image 1 from sample.pdf. vision caption: Architecture diagram showing api gateway worker queue database."]
    candidate_metas = [{
        "source_fingerprint": "src1",
        "page_number": "3",
        "source": "sample.pdf",
        "modality": "image",
    }]
    candidate_ids = ["image-1"]

    docs, metas, ids = rag_tools._dedupe_multimodal_payloads(
        base_docs,
        base_metas,
        candidate_docs,
        candidate_metas,
        candidate_ids,
    )

    assert docs == candidate_docs
    assert metas == candidate_metas
    assert ids == candidate_ids


def test_truncate_to_budget_skips_oversized_chunks():
    kept, diagnostics = rag_tools._truncate_to_budget(
        ranked=[
            ("x" * 200, {"source": "large.txt"}, "large-id"),
            ("small chunk", {"source": "small.txt"}, "small-id"),
        ],
        max_tokens=20,
        return_details=True,
    )

    assert kept == [("small chunk", {"source": "small.txt"}, "small-id")]
    assert diagnostics["skipped_count"] == 1
    assert diagnostics["skipped_chunks"][0]["id"] == "large-id"


def test_rag_retrieve_records_query_diagnostics(monkeypatch):
    class _DummyCollection:
        def count(self):
            return 1

    monkeypatch.setattr(rag_tools, "_list_rag_collections", lambda: [_DummyCollection()])
    monkeypatch.setattr(
        rag_tools,
        "_query_all_matching_collections",
        lambda **_kwargs: (
            ["strong signal", "backup evidence"],
            [{"source": "alpha.txt"}, {"source": "beta.txt"}],
            ["id-a", "id-b"],
        ),
    )

    def _fake_rerank_chunks(**_kwargs):
        return (
            [("strong signal", {"source": "alpha.txt"}, "id-a")],
            {
                "candidate_count": 2,
                "rerank_count": 2,
                "filtered_count": 1,
                "selected_files": ["alpha.txt"],
                "file_scores": {"alpha.txt": 0.95},
            },
        )

    monkeypatch.setattr(rag_tools, "_rerank_chunks", _fake_rerank_chunks)

    output = rag_tools.rag_retrieve.invoke(
        {
            "query": "example",
            "project": "Diag Project",
            "top_k": 1,
            "mode": "Top-K Globally",
            "fetch_k": 10,
            "max_files": 2,
        }
    )
    diagnostics = rag_tools.get_last_rag_query_diagnostics()

    assert output.startswith("Retrieved 1 chunk")
    assert diagnostics["status"] == "ok"
    assert diagnostics["project"] == "Diag Project"
    assert diagnostics["final_chunk_ids"] == ["id-a"]
    assert diagnostics["selected_files"] == ["alpha.txt"]


def test_rag_retrieve_applies_modality_filter(monkeypatch):
    class _DummyCollection:
        def count(self):
            return 1

    observed: dict[str, object] = {}

    monkeypatch.setattr(rag_tools, "_list_rag_collections", lambda: [_DummyCollection()])

    def _fake_query_all_matching_collections(**kwargs):
        observed["where_filter"] = kwargs.get("where_filter")
        return (
            ["image chunk"],
            [{"source": "diagram.pdf", "modality": "image"}],
            ["img-1"],
        )

    monkeypatch.setattr(rag_tools, "_query_all_matching_collections", _fake_query_all_matching_collections)
    monkeypatch.setattr(
        rag_tools,
        "_rerank_chunks",
        lambda **_kwargs: ([("image chunk", {"source": "diagram.pdf", "modality": "image"}, "img-1")], {
            "candidate_count": 1,
            "rerank_count": 1,
            "filtered_count": 1,
            "selected_files": ["diagram.pdf"],
            "file_scores": {"diagram.pdf": 0.9},
        }),
    )

    output = rag_tools.rag_retrieve.invoke(
        {
            "query": "show me the diagram",
            "project": "Proj",
            "modalities": "image",
            "top_k": 1,
            "mode": "Top-K Globally",
            "fetch_k": 10,
            "max_files": 2,
        }
    )

    assert output.startswith("Retrieved 1 chunk")
    assert observed["where_filter"] == {
        "$and": [
            {"project": {"$eq": "Proj"}},
            {"modality": {"$in": ["image"]}},
        ]
    }


def test_extract_table_candidates_rejects_outline_bullets():
    page_text = "\n".join(
        [
            "○ 1.4.1 Services (5 mins): Chatbots, virtual assistants, automated support.",
            "○ 1.4.2 Marketing & Sales (5 mins): Personalized recommendations, lead scoring.",
        ]
    )

    assert rag_tools._extract_table_candidates(page_text) == []


def test_extract_table_candidates_accepts_columnar_rows():
    page_text = "\n".join(
        [
            "Category    Q1    Q2    Q3",
            "Revenue     10    12    15",
            "Costs       4     5     6",
        ]
    )

    results = rag_tools._extract_table_candidates(page_text)

    assert len(results) == 1
    assert "Category    Q1    Q2    Q3" in results[0]
    assert "Revenue     10    12    15" in results[0]


def test_extract_pdf_table_records_marks_table_modality():
    page = SimpleNamespace(
        page_content="\n".join(
            [
                "Metric    Jan    Feb",
                "Sales     20     25",
                "Profit    5      7",
            ]
        ),
        metadata={"page_number": "1"},
    )

    docs_payload, metas_payload, ids_payload = rag_tools._extract_pdf_table_records(
        file_path=SimpleNamespace(name="sample.pdf"),
        pages=[page],
        project_name="Proj",
        theme_name="Theme",
        source_fingerprint="fingerprint123",
        date_added="2026-04-26",
        doc_meta={},
    )

    assert len(docs_payload) == 1
    assert len(metas_payload) == 1
    assert len(ids_payload) == 1
    assert metas_payload[0]["modality"] == "table"
    assert metas_payload[0]["page_number"] == "1"
    assert metas_payload[0]["asset_path"].endswith("page_1_table_1.txt")


def test_extract_pdf_table_records_prefers_pymupdf_tables(monkeypatch):
    class _FakeTable:
        def extract(self):
            return [
                ["Metric", "Jan", "Feb"],
                ["Sales", "20", "25"],
            ]

    class _FakeFinder:
        tables = [_FakeTable()]

    class _FakePdfPage:
        def find_tables(self):
            return _FakeFinder()

    class _FakePdf:
        def __len__(self):
            return 1

        def load_page(self, index):
            assert index == 0
            return _FakePdfPage()

        def close(self):
            return None

    import fitz

    monkeypatch.setattr(fitz, "open", lambda _path: _FakePdf())

    page = SimpleNamespace(
        page_content="Narrative text only, not a table fallback candidate.",
        metadata={"page_number": "1"},
    )

    docs_payload, metas_payload, ids_payload = rag_tools._extract_pdf_table_records(
        file_path=SimpleNamespace(name="sample.pdf", __str__=lambda self: "sample.pdf"),
        pages=[page],
        project_name="Proj",
        theme_name="Theme",
        source_fingerprint="fingerprint456",
        date_added="2026-04-26",
        doc_meta={},
    )

    assert len(docs_payload) == 1
    assert len(metas_payload) == 1
    assert len(ids_payload) == 1
    assert "Metric Jan Feb Sales 20 25" in docs_payload[0]
    assert metas_payload[0]["table_extraction_method"] == "pymupdf"


def test_extract_pdf_table_records_falls_back_to_heuristic_when_no_parser_table(monkeypatch):
    class _FakeFinder:
        tables = []

    class _FakePdfPage:
        def find_tables(self):
            return _FakeFinder()

    class _FakePdf:
        def __len__(self):
            return 1

        def load_page(self, index):
            assert index == 0
            return _FakePdfPage()

        def close(self):
            return None

    import fitz

    monkeypatch.setattr(fitz, "open", lambda _path: _FakePdf())

    page = SimpleNamespace(
        page_content="\n".join(
            [
                "Metric    Jan    Feb",
                "Sales     20     25",
                "Profit    5      7",
            ]
        ),
        metadata={"page_number": "1"},
    )

    docs_payload, metas_payload, ids_payload = rag_tools._extract_pdf_table_records(
        file_path=SimpleNamespace(name="sample.pdf", __str__=lambda self: "sample.pdf"),
        pages=[page],
        project_name="Proj",
        theme_name="Theme",
        source_fingerprint="fingerprint789",
        date_added="2026-04-26",
        doc_meta={},
    )

    assert len(docs_payload) == 1
    assert len(metas_payload) == 1
    assert len(ids_payload) == 1
    assert metas_payload[0]["table_extraction_method"] == "heuristic"


def test_rag_prompt_no_longer_mentions_write_todos():
    assert "write_todos" not in RAG_SUB_INSTRUCTIONS


def test_rag_prompt_uses_slide_outline_handoff_contract():
    assert "slide-ready outline" in RAG_SUB_INSTRUCTIONS
    assert "never own final PPTX generation" in RAG_SUB_INSTRUCTIONS
    assert "create a presentation file" not in RAG_SUB_INSTRUCTIONS


def test_ragsub_toolset_does_not_include_generate_presentation():
    tool_names = {tool.name for tool in agent.SUBAGENT_TOOL_MAP["ragsub"]}

    assert "generate_presentation" not in tool_names


def test_generate_chunk_context_marks_empty_response(monkeypatch):
    class _EmptyResponseLlm:
        def invoke(self, _prompt):
            return SimpleNamespace(content=[])

    monkeypatch.setattr(rag_tools, "_get_context_llm", lambda: _EmptyResponseLlm())
    rag_tools._context_llm_status["model"] = "gemma4:26b"

    result = rag_tools._generate_chunk_context("full doc", "chunk text")

    assert result == ""
    assert rag_tools._context_llm_status["state"] == "empty-response"
    assert "returned an empty response" in rag_tools._context_llm_status["error"]
