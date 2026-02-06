import json
import requests
from app.core.config import settings

ROUTER_PROMPT = """
You are a document router. Classify the document type and return JSON only.
Document types enum:
- rfq_quote_po
- contract_tech_agreement
- bom_parts_list
- drawings_revision
- meeting_minutes
- project_schedule
- debug_test_report
- inspection_report_ncr
- fat_sat_report
- delivery_commissioning
- unknown
Return JSON: {"doc_type": "<enum>", "confidence": 0.0}
"""

EXTRACTOR_PROMPT = """
Extract structured data. Return JSON only, matching schema:
{
  "organizations": [{"name": "...", "type": "customer|supplier|internal", "aliases": []}],
  "people": [{"name":"...", "role": "optional"}],
  "references": [{"type":"project|po|quote|contract", "value":"..."}],
  "items": [{"name":"...", "part_no":"...", "qty":null, "unit_price":null, "total":null}],
  "events": [{"type":"meeting|inspection|test|shipment|fat|sat", "date":null, "result":null, "location":null}],
  "issues": [{"description":"...", "severity":"low|medium|high", "owner":null, "due_date":null}],
  "actions": [{"task":"...", "owner":null, "due_date":null}],
  "project": {"project_code":null, "name":null, "milestones":[]},
  "quality": {"inspection_result":null, "ncr": {"description":null, "root_cause":null, "corrective_action":null}}
}
"""


def parse_json_response(raw: str) -> dict:
    return json.loads(raw)


def _ollama_generate(prompt: str, text: str) -> dict:
    payload = {
        "model": settings.ollama_model,
        "prompt": f"{prompt}\n\nDocument:\n{text}",
        "stream": False,
        "options": {"temperature": 0.2},
    }
    response = requests.post(f"{settings.ollama_url}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    raw = data.get("response", "{}").strip()
    try:
        return parse_json_response(raw)
    except json.JSONDecodeError:
        repair_payload = payload.copy()
        repair_payload["prompt"] = (
            "Return valid JSON only. Fix formatting.\n" + repair_payload["prompt"]
        )
        response = requests.post(
            f"{settings.ollama_url}/api/generate", json=repair_payload, timeout=120
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "{}").strip()
        return parse_json_response(raw)


def route_document(text: str) -> dict:
    return _ollama_generate(ROUTER_PROMPT, text)


def extract_document(text: str, doc_type: str) -> dict:
    prompt = f"{EXTRACTOR_PROMPT}\nDocument type: {doc_type}"
    return _ollama_generate(prompt, text)
