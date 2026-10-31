"""Offline biometric evaluation runs (FAR / FRR / EER on labelled data)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from backend.core.deps import Actor, client_ip, require
from backend.core.permissions import Permission
from backend.database import get_db
from backend.models.documents import EvaluationRun
from backend.services import evaluation_service
from backend.services.audit_service import audit
from backend.services.inference import runtime
from backend.services.policy_service import APPROVED_RECOGNITION_MODELS, get_active_policy

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


def run_view(r: EvaluationRun) -> dict:
    return {"id": r.id, "status": r.status, "model_name": r.model_name, "dataset_label": r.dataset_label,
            "created_at": r.created_at, "completed_at": r.completed_at, "identity_count": r.identity_count,
            "image_count": r.image_count, "skipped_images": r.skipped_images, "genuine_pairs": r.genuine_pairs,
            "impostor_pairs": r.impostor_pairs, "error": r.error,
            "results": json.loads(r.results_json) if r.results_json else None}


@router.get("/runs")
def list_runs(db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.ANALYTICS_READ))):
    rows = (db.query(EvaluationRun).filter(EvaluationRun.organization_id == actor.org_id)
            .order_by(EvaluationRun.created_at.desc()).limit(50).all())
    return {"runs": [run_view(r) for r in rows], "approved_models": list(APPROVED_RECOGNITION_MODELS)}


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.ANALYTICS_READ))):
    r = db.query(EvaluationRun).filter(EvaluationRun.id == run_id, EvaluationRun.organization_id == actor.org_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run_view(r)


@router.post("/runs", status_code=202)
async def create_run(request: Request, dataset: UploadFile = File(...), label: str = Form("", max_length=200),
                     db: Session = Depends(get_db), actor: Actor = Depends(require(Permission.EVALUATION_RUN))):
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="Models are not ready yet")
    data = await dataset.read(evaluation_service.MAX_ZIP_BYTES + 1)
    try:
        groups = evaluation_service.parse_dataset(data)
    except evaluation_service.EvaluationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    policy = get_active_policy(db, actor.org_id)
    run = EvaluationRun(organization_id=actor.org_id, created_by_user_id=actor.id, model_name=runtime.provider.recognition_model,
                        dataset_label=label.strip() or dataset.filename, status="RUNNING",
                        identity_count=len(groups), image_count=sum(len(v) for v in groups.values()))
    db.add(run)
    db.commit()
    audit(db, actor.id, "EVALUATION_STARTED", "evaluation_run", run.id,
          {"identities": run.identity_count, "images": run.image_count}, client_ip(request), org_id=actor.org_id)
    evaluation_service.start_evaluation(run.id, groups, float(policy.match_threshold))
    return run_view(run)
