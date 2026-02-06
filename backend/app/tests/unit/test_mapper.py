from app.services.mapper import map_to_proposals


def test_map_to_proposals_creates_action_and_issue():
    text = "Action: Engineer to verify sensor wiring. Issue: Conveyor vibration observed."
    data = {
        "project": {"project_code": "AL-1001", "name": "Automation Line", "milestones": []},
        "organizations": [{"name": "Nanjing Aline Automation Company", "type": "customer", "aliases": []}],
        "actions": [{"task": "Verify sensor wiring", "owner": None, "due_date": None}],
        "issues": [{"description": "Conveyor vibration observed", "severity": "high", "owner": None, "due_date": None}],
        "quality": {"inspection_result": None, "ncr": {"description": None, "root_cause": None, "corrective_action": None}},
    }
    proposals = map_to_proposals(text, "sample.txt", data)
    targets = {proposal["target_table"] for proposal in proposals}
    assert "projects" in targets
    assert "customers" in targets
    assert "tasks" in targets
    assert "issues" in targets
