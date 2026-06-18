from ragsub_agent import tools as rag_tools


class _DummyCollection:
    def count(self) -> int:
        return 1


def test_rag_retrieve_output_includes_references(monkeypatch):
    docs = ["Grounded chunk text about asyncio locks."]
    metas = [{
        "source": "asyncio_primitives.txt",
        "section": "Concurrency",
        "topics": "asyncio, locks",
        "page_number": "2",
        "doc_title": "Asyncio coordination notes",
        "doc_authors": "Smith",
        "doc_year": "2024",
    }]
    ids = ["chunk-1"]

    monkeypatch.setattr(
        rag_tools,
        "_list_rag_collections",
        lambda: [_DummyCollection()],
    )
    monkeypatch.setattr(
        rag_tools,
        "_query_all_matching_collections",
        lambda query, n_results, where_filter=None: (docs, metas, ids),
    )
    monkeypatch.setattr(
        rag_tools,
        "_rerank_chunks",
        lambda **kwargs: [(docs[0], metas[0], ids[0])],
    )
    monkeypatch.setattr(
        rag_tools,
        "_truncate_to_budget",
        lambda ranked, max_tokens: ranked,
    )

    output = rag_tools.rag_retrieve.invoke({
        "query": "Which asyncio primitive guards shared mutable state?",
        "project": "RAG_EVAL_PRIMARY",
        "themes": "qa-primary",
        "top_k": 3,
        "mode": "Top-K Globally",
        "fetch_k": 10,
        "max_files": 5,
    })

    assert "[R1] asyncio_primitives.txt p.2 § Concurrency" in output
    assert "References:" in output
    assert "(Smith, 2024, p. 2)" in output
