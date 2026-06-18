from message_sanitizer import coerce_message_content_to_text, sanitize_history_pairs


def test_coerce_message_content_to_text_flattens_file_blocks():
    content = [
        {"type": "text", "text": "Draft a short bio based on this file."},
        {"type": "file", "filename": "Ven Seyhah bio.docx"},
    ]

    result = coerce_message_content_to_text(content)

    assert "Draft a short bio based on this file." in result
    assert "[Attached file: Ven Seyhah bio.docx]" in result


def test_sanitize_history_pairs_converts_non_string_content():
    history = [
        ("user", [{"type": "text", "text": "Hello"}, {"type": "image"}]),
        ("assistant", {"type": "text", "text": "Hi"}),
    ]

    result = sanitize_history_pairs(history)

    assert result == [
        ("user", "Hello\n[Attached image]"),
        ("assistant", "Hi"),
    ]


def test_coerce_message_content_to_text_extracts_local_text_file(tmp_path):
    sample = tmp_path / "note.txt"
    sample.write_text("Line one\nLine two", encoding="utf-8")

    content = [
        {"type": "text", "text": "Please read this attachment."},
        {"type": "file", "filename": "note.txt", "path": str(sample)},
    ]

    result = coerce_message_content_to_text(content)

    assert "Please read this attachment." in result
    assert "[Attached file: note.txt]" in result
    assert "Line one" in result
