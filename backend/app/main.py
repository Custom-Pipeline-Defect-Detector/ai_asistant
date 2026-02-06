from fastapi import FastAPI
from app.api import auth, documents, proposals, dashboard, projects
from app.db.session import SessionLocal
from app.models.models import Role, User
from app.core.security import hash_password

app = FastAPI(title="Aline AI Assistant")

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(proposals.router, prefix="/api/proposals", tags=["proposals"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])


@app.on_event("startup")
def seed_admin():
    db = SessionLocal()
    try:
        role_names = ["Admin", "Manager", "PM", "Sales", "Engineer", "Technician", "QC", "Viewer"]
        existing_roles = {role.name for role in db.query(Role).all()}
        for name in role_names:
            if name not in existing_roles:
                db.add(Role(name=name))
        db.commit()
        admin = db.query(User).filter_by(email="admin@aline.local").first()
        if not admin:
            admin = User(
                email="admin@aline.local",
                name="Admin",
                password_hash=hash_password("Admin123!"),
                is_active=True,
            )
            admin.roles = db.query(Role).filter(Role.name == "Admin").all()
            db.add(admin)
            db.commit()
    finally:
        db.close()
