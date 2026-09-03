from teacup_agent.memory import Memory


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


# --- notes: reflect.py's lower-trust tier, separate from remember()'s facts ------


def test_note_dedupes_and_evicts_like_facts_do(tmp_path):
    m = Memory(tmp_path / "memory.json", note_limit=2)
    m.note("experience", "a")
    m.note("experience", "a")  # a duplicate must not be stored twice
    m.note("lesson", "b")
    m.note("experience", "c")  # evicts "a", the oldest
    assert m.notes == [{"kind": "lesson", "text": "b"}, {"kind": "experience", "text": "c"}]


def test_notes_survive_reload(tmp_path):
    path = tmp_path / "memory.json"
    Memory(path).note("experience", "searching once was enough")
    assert Memory(path).notes == [{"kind": "experience", "text": "searching once was enough"}]


def test_recall_labels_notes_separately_and_after_facts(tmp_path):
    m = Memory(tmp_path / "memory.json")
    m.remember("fact one")
    m.note("lesson", "lesson one")
    text = m.recall()
    assert "fact one" in text and "lesson one" in text
    assert "Unreviewed notes" in text
    # Facts come first: they are the model's own, notes are the harness's afterthought.
    assert text.index("You remember these facts") < text.index("Unreviewed notes")


def test_recall_omits_the_notes_block_when_there_are_none(tmp_path):
    m = Memory(tmp_path / "memory.json")
    m.remember("fact one")
    assert "Unreviewed notes" not in m.recall()
