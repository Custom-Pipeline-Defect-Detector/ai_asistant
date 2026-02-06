from datetime import datetime, date
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserBase(BaseModel):
    id: int
    email: str
    name: str

    class Config:
        from_attributes = True


class DocumentCreate(BaseModel):
    folder_path_hint: Optional[str] = None


class DocumentOut(BaseModel):
    id: int
    filename: str
    mime: str
    storage_path: str
    file_hash: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentVersionOut(BaseModel):
    id: int
    doc_id: int
    version: int
    extracted_text_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ProposalOut(BaseModel):
    id: int
    doc_version_id: int
    proposed_action: str
    target_module: str
    target_table: str
    target_entity_id: Optional[int]
    proposed_fields: Dict[str, Any]
    field_confidence: Dict[str, Any]
    evidence: Dict[str, Any]
    questions: Dict[str, Any]
    status: str
    created_at: datetime
    reviewed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ProposalUpdate(BaseModel):
    proposed_fields: Dict[str, Any]


class ProposalDecision(BaseModel):
    decision: str
    reason: Optional[str] = None


class DashboardCounts(BaseModel):
    projects: int
    tasks: int
    pending_proposals: int
    ncrs: int


class ProjectOut(BaseModel):
    id: int
    project_code: str
    name: str
    status: str
    start_date: Optional[date]
    due_date: Optional[date]

    class Config:
        from_attributes = True
