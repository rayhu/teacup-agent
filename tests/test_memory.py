from mini_agent.memory import Memory


def test_memory_roundtrip_and_dedupe(tmp_path):
    path = tmp_path / "memory.json"
    m = Memory(path)
    m.remember("user prefers concise answers")
    m.remember("user prefers concise answers")  # a duplicate must not be stored twice
    assert m.facts == ["user prefers concise answers"]

    reloaded = Memory(path)
    assert reloaded.facts == ["user prefers concise answers"]
    assert "user prefers concise answers" in reloaded.recall()


def test_corrupted_memory_file_does_not_crash(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ not json", encoding="utf-8")
    assert Memory(path).facts == []
