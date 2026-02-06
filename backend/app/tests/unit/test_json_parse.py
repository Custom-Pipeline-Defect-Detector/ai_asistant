from app.services.ollama import parse_json_response


def test_parse_json_response():
    raw = '{"doc_type": "meeting_minutes", "confidence": 0.8}'
    data = parse_json_response(raw)
    assert data["doc_type"] == "meeting_minutes"
