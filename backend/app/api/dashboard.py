from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.deps import get_db, require_roles
from app.models.models import Project, Task, Proposal, NCR
from app.schemas.schemas import DashboardCounts

router = APIRouter()


@router.get("/counts", response_model=DashboardCounts)
def get_counts(
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "QC", "Viewer")),
):
    return DashboardCounts(
        projects=db.query(Project).count(),
        tasks=db.query(Task).count(),
        pending_proposals=db.query(Proposal).filter_by(status="pending").count(),
        ncrs=db.query(NCR).count(),
    )
