from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.deps import get_db, require_roles
from app.models.models import Project
from app.schemas.schemas import ProjectOut

router = APIRouter()


@router.get("/", response_model=list[ProjectOut])
def list_projects(
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "Engineer", "Technician", "QC", "Viewer")),
):
    return db.query(Project).order_by(Project.created_at.desc()).all()


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "Engineer", "Technician", "QC", "Viewer")),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
