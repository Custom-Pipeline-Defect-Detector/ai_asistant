import datetime as dt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.deps import get_db, require_roles
from app.models.models import (
    Proposal,
    AuditLog,
    Customer,
    Project,
    Task,
    Issue,
    NCR,
)
from app.schemas.schemas import ProposalOut, ProposalUpdate, ProposalDecision

router = APIRouter()


def _apply_proposal(db: Session, proposal: Proposal, user_id: int) -> None:
    fields = proposal.proposed_fields
    if proposal.target_table == "customers":
        entity = Customer(name=fields.get("name"), aliases=fields.get("aliases", []))
        db.add(entity)
        db.flush()
        db.add(
            AuditLog(
                actor_user_id=user_id,
                action="approve",
                entity_table="customers",
                entity_id=entity.id,
                before=None,
                after=fields,
            )
        )
    elif proposal.target_table == "projects":
        customer_id = fields.get("customer_id")
        if not customer_id:
            customer = db.query(Customer).first()
            if not customer:
                customer = Customer(name="Unknown Customer", aliases=[])
                db.add(customer)
                db.flush()
            customer_id = customer.id
        entity = Project(
            project_code=fields.get("project_code"),
            name=fields.get("name"),
            customer_id=customer_id,
            status=fields.get("status", "pending"),
        )
        db.add(entity)
        db.flush()
        db.add(
            AuditLog(
                actor_user_id=user_id,
                action="approve",
                entity_table="projects",
                entity_id=entity.id,
                before=None,
                after=fields,
            )
        )
    elif proposal.target_table == "tasks":
        project_id = fields.get("project_id")
        if not project_id:
            project = db.query(Project).first()
            if not project:
                raise HTTPException(status_code=400, detail="Missing project for task")
            project_id = project.id
        entity = Task(
            project_id=project_id,
            title=fields.get("title"),
            description=fields.get("description"),
            status=fields.get("status", "open"),
        )
        db.add(entity)
        db.flush()
        db.add(
            AuditLog(
                actor_user_id=user_id,
                action="approve",
                entity_table="tasks",
                entity_id=entity.id,
                before=None,
                after=fields,
            )
        )
    elif proposal.target_table == "issues":
        project_id = fields.get("project_id")
        if not project_id:
            project = db.query(Project).first()
            if not project:
                raise HTTPException(status_code=400, detail="Missing project for issue")
            project_id = project.id
        entity = Issue(
            project_id=project_id,
            severity=fields.get("severity", "low"),
            description=fields.get("description"),
            status=fields.get("status", "open"),
        )
        db.add(entity)
        db.flush()
        db.add(
            AuditLog(
                actor_user_id=user_id,
                action="approve",
                entity_table="issues",
                entity_id=entity.id,
                before=None,
                after=fields,
            )
        )
    elif proposal.target_table == "ncrs":
        project_id = fields.get("project_id")
        if not project_id:
            project = db.query(Project).first()
            if not project:
                raise HTTPException(status_code=400, detail="Missing project for NCR")
            project_id = project.id
        entity = NCR(
            project_id=project_id,
            description=fields.get("description"),
            root_cause=fields.get("root_cause"),
            corrective_action=fields.get("corrective_action"),
            status=fields.get("status", "open"),
        )
        db.add(entity)
        db.flush()
        db.add(
            AuditLog(
                actor_user_id=user_id,
                action="approve",
                entity_table="ncrs",
                entity_id=entity.id,
                before=None,
                after=fields,
            )
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported target table")


@router.get("/", response_model=list[ProposalOut])
def list_proposals(
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "QC")),
):
    return db.query(Proposal).order_by(Proposal.created_at.desc()).all()


@router.get("/{proposal_id}", response_model=ProposalOut)
def get_proposal(
    proposal_id: int,
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "QC")),
):
    proposal = db.get(Proposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


@router.put("/{proposal_id}", response_model=ProposalOut)
def update_proposal(
    proposal_id: int,
    payload: ProposalUpdate,
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "QC")),
):
    proposal = db.get(Proposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    proposal.proposed_fields = payload.proposed_fields
    db.commit()
    db.refresh(proposal)
    return proposal


@router.post("/{proposal_id}/decision", response_model=ProposalOut)
def decide_proposal(
    proposal_id: int,
    payload: ProposalDecision,
    db: Session = Depends(get_db),
    user=Depends(require_roles("Admin", "Manager", "PM", "QC")),
):
    proposal = db.get(Proposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if proposal.status != "pending":
        raise HTTPException(status_code=400, detail="Proposal already processed")

    if payload.decision == "approve":
        _apply_proposal(db, proposal, user.id)
        proposal.status = "approved"
        proposal.reviewed_at = dt.datetime.utcnow()
        proposal.reviewer_id = user.id
    elif payload.decision == "reject":
        proposal.status = "rejected"
        proposal.questions = {"rejected_reason": payload.reason}
        proposal.reviewed_at = dt.datetime.utcnow()
        proposal.reviewer_id = user.id
        db.add(
            AuditLog(
                actor_user_id=user.id,
                action="reject",
                entity_table="proposals",
                entity_id=proposal.id,
                before=None,
                after={"reason": payload.reason},
            )
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid decision")

    db.commit()
    db.refresh(proposal)
    return proposal
