from typing import Any, Dict, List
from app.services.extraction import split_snippet


def _evidence_for_field(text: str, filename: str, value: str) -> Dict[str, Any]:
    snippet, location = split_snippet(text, value)
    return {"snippet": snippet, "location": location, "source": filename}


def map_to_proposals(text: str, filename: str, intermediate: Dict[str, Any]) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []

    project = intermediate.get("project") or {}
    if project.get("project_code") and project.get("name"):
        proposals.append(
            {
                "proposed_action": "create",
                "target_module": "projects",
                "target_table": "projects",
                "proposed_fields": {
                    "project_code": project.get("project_code"),
                    "name": project.get("name"),
                },
                "field_confidence": {"project_code": 0.7, "name": 0.7},
                "evidence": {
                    "project_code": _evidence_for_field(
                        text, filename, project.get("project_code")
                    ),
                    "name": _evidence_for_field(text, filename, project.get("name")),
                },
                "questions": {},
            }
        )

    orgs = intermediate.get("organizations") or []
    for org in orgs:
        if org.get("type") == "customer" and org.get("name"):
            proposals.append(
                {
                    "proposed_action": "create",
                    "target_module": "customers",
                    "target_table": "customers",
                    "proposed_fields": {"name": org.get("name"), "aliases": org.get("aliases") or []},
                    "field_confidence": {"name": 0.6, "aliases": 0.4},
                    "evidence": {
                        "name": _evidence_for_field(text, filename, org.get("name")),
                        "aliases": _evidence_for_field(text, filename, org.get("name")),
                    },
                    "questions": {},
                }
            )

    actions = intermediate.get("actions") or []
    for action in actions:
        if action.get("task"):
            proposals.append(
                {
                    "proposed_action": "create",
                    "target_module": "tasks",
                    "target_table": "tasks",
                    "proposed_fields": {"title": action.get("task"), "description": action.get("task")},
                    "field_confidence": {"title": 0.5, "description": 0.5},
                    "evidence": {
                        "title": _evidence_for_field(text, filename, action.get("task")),
                        "description": _evidence_for_field(text, filename, action.get("task")),
                    },
                    "questions": {},
                }
            )

    issues = intermediate.get("issues") or []
    for issue in issues:
        if issue.get("description"):
            proposals.append(
                {
                    "proposed_action": "create",
                    "target_module": "issues",
                    "target_table": "issues",
                    "proposed_fields": {
                        "description": issue.get("description"),
                        "severity": issue.get("severity", "low"),
                    },
                    "field_confidence": {"description": 0.6, "severity": 0.4},
                    "evidence": {
                        "description": _evidence_for_field(
                            text, filename, issue.get("description")
                        ),
                        "severity": _evidence_for_field(
                            text, filename, issue.get("severity", "low")
                        ),
                    },
                    "questions": {},
                }
            )

    quality = intermediate.get("quality") or {}
    ncr = quality.get("ncr") or {}
    if ncr.get("description"):
        proposals.append(
            {
                "proposed_action": "create",
                "target_module": "ncrs",
                "target_table": "ncrs",
                "proposed_fields": {
                    "description": ncr.get("description"),
                    "root_cause": ncr.get("root_cause"),
                    "corrective_action": ncr.get("corrective_action"),
                },
                "field_confidence": {
                    "description": 0.7,
                    "root_cause": 0.4,
                    "corrective_action": 0.4,
                },
                "evidence": {
                    "description": _evidence_for_field(text, filename, ncr.get("description")),
                    "root_cause": _evidence_for_field(text, filename, ncr.get("root_cause") or ""),
                    "corrective_action": _evidence_for_field(
                        text, filename, ncr.get("corrective_action") or ""
                    ),
                },
                "questions": {},
            }
        )

    return proposals
