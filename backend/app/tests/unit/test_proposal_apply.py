from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from app.models.models import Proposal, User, Role
from app.api.proposals import _apply_proposal
from app.core.security import hash_password


def setup_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_apply_proposal_creates_project_audit_log():
    db = setup_db()
    role = Role(name="Admin")
    user = User(email="test@example.com", name="Tester", password_hash=hash_password("pass"))
    user.roles.append(role)
    db.add_all([role, user])
    db.commit()

    proposal = Proposal(
        doc_version_id=1,
        proposed_action="create",
        target_module="projects",
        target_table="projects",
        proposed_fields={"project_code": "AL-1001", "name": "Automation Line"},
        field_confidence={"project_code": 0.7, "name": 0.7},
        evidence={"project_code": {}, "name": {}},
        questions={},
        status="pending",
    )
    db.add(proposal)
    db.commit()

    _apply_proposal(db, proposal, user.id)
    db.commit()
    projects = db.execute("SELECT count(*) FROM projects").scalar()
    assert projects == 1
