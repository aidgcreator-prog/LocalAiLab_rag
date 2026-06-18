from literature_review import tools as literature_tools
from ragsub_agent import tools as rag_tools


class _FakeCollection:
    def __init__(self, ids, metadatas):
        self._ids = list(ids)
        self._metadatas = list(metadatas)
        self.deleted_ids = []

    def count(self):
        return len(self._ids)

    def peek(self, _count):
        return {
            "ids": list(self._ids),
            "metadatas": list(self._metadatas),
        }

    def delete(self, ids):
        self.deleted_ids.extend(ids)

    def get(self, include=None):
        return {"metadatas": list(self._metadatas)}


def test_delete_rag_documents_stays_within_project(monkeypatch):
    alpha = _FakeCollection(
        ids=["alpha-1", "alpha-2"],
        metadatas=[
            {"source": "shared.txt", "theme": "alpha"},
            {"source": "other.txt", "theme": "alpha"},
        ],
    )
    beta = _FakeCollection(
        ids=["beta-1"],
        metadatas=[
            {"source": "shared.txt", "theme": "beta"},
        ],
    )

    collections = {"Alpha": alpha, "Beta": beta}
    monkeypatch.setattr(
        rag_tools,
        "_get_vector_collection",
        lambda project="Default": collections[project],
    )

    deleted = rag_tools.delete_rag_documents(project="Alpha", source="shared.txt")

    assert deleted == 1
    assert alpha.deleted_ids == ["alpha-1"]
    assert beta.deleted_ids == []


def test_get_rag_index_summary_uses_single_project_collection(monkeypatch):
    alpha = _FakeCollection(
        ids=["alpha-1"],
        metadatas=[
            {
                "source": "shared.txt",
                "theme": "alpha",
                "date_added": "2026-04-19",
            }
        ],
    )
    beta = _FakeCollection(
        ids=["beta-1"],
        metadatas=[
            {
                "source": "shared.txt",
                "theme": "beta",
                "date_added": "2026-04-20",
            }
        ],
    )

    collections = {"Alpha": alpha, "Beta": beta}
    monkeypatch.setattr(
        rag_tools,
        "_get_vector_collection",
        lambda project="Default": collections[project],
    )

    summary = rag_tools.get_rag_index_summary(project="Alpha")

    assert summary["project"] == "Alpha"
    assert summary["total_chunks"] == 1
    assert sorted(summary["files"].keys()) == ["shared.txt"]
    assert summary["files"]["shared.txt"]["themes"] == {"alpha"}


def test_generate_literature_report_reads_project_scoped_collection(monkeypatch, tmp_path):
    seen_projects = []

    def _fake_get_vector_collection(project="Default"):
        seen_projects.append(project)
        return _FakeCollection(
            ids=["paper-1"],
            metadatas=[
                {
                    "source": "paper-one.pdf",
                    "doc_title": "Paper One",
                    "doc_authors": "Smith; Jones",
                    "doc_year": "2024",
                    "doi": "10.1000/example",
                    "journal": "Journal of Testing",
                    "paper_abstract": "Abstract",
                    "citation_count": "7",
                    "pdf_method": "pdf",
                }
            ],
        )

    def _fake_generate_docx_report(
        project,
        query,
        synthesis_text,
        paper_records,
        output_path,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ok")
        assert project == "Project Alpha"
        assert query == "example query"
        assert synthesis_text == "## Findings"
        assert len(paper_records) == 1
        return output_path

    monkeypatch.setattr(literature_tools, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "ragsub_agent.tools.get_vector_collection",
        _fake_get_vector_collection,
    )
    monkeypatch.setattr(
        "literature_review.report_generator.generate_docx_report",
        _fake_generate_docx_report,
    )

    result = literature_tools.generate_literature_report.invoke(
        {
            "project": "Project Alpha",
            "query": "example query",
            "synthesis_text": "## Findings",
        }
    )

    assert seen_projects == ["Project Alpha"]
    assert "[OK] Report generated:" in result
