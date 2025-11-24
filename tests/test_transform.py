import json
from pathlib import Path
from src.transform import process_file

def test_process_sample(tmp_path):
    sample = tmp_path / "sample.jsonl"
    out = tmp_path / "out.jsonl"
    # create minimal sample
    obj = {
        "id": "HADOOP-1",
        "title": "Test issue",
        "text": "This is a long enough description to pass min len",
        "created": "2020-01-01T00:00:00.000+0000"
    }
    sample.write_text(json.dumps(obj) + "\n", encoding="utf8")
    wrote = process_file(str(sample), str(out), min_text_len=10)
    assert wrote == 1
    data = out.read_text(encoding="utf8").strip().splitlines()
    assert len(data) == 1
    parsed = json.loads(data[0])
    assert parsed["id"] == "HADOOP-1"
    assert "created" in parsed
