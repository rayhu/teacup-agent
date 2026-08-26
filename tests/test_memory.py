from mini_agent.memory import Memory


def test_memory_roundtrip_and_dedupe(tmp_path):
    path = tmp_path / "memory.json"
    m = Memory(path)
    m.remember("用户偏好中文回答")
    m.remember("用户偏好中文回答")  # 重复不应写两次
    assert m.facts == ["用户偏好中文回答"]

    reloaded = Memory(path)
    assert reloaded.facts == ["用户偏好中文回答"]
    assert "用户偏好中文回答" in reloaded.recall()


def test_corrupted_memory_file_does_not_crash(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ not json", encoding="utf-8")
    assert Memory(path).facts == []
