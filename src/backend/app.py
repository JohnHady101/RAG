"""FastAPI backend: ask questions over the RAG knowledge base.

Run from the repo root:

    uvicorn src.backend.app:app --reload
    # or
    python src/backend/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from qa import answer_question, retrieve

app = FastAPI(title="RAG API")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class Source(BaseModel):
    id: int
    content: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        answer = answer_question(req.question, top_k=req.top_k)
        rows = retrieve(req.question, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        answer=answer,
        sources=[Source(id=i, content=c, score=s) for i, c, _m, s in rows],
    )


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

if os.path.isdir(STATIC_DIR):
    app.mount(
        "/",
        StaticFiles(
            directory=STATIC_DIR,
            html=True,
        ),
        name="static",
    )

# uvicorn app:app --host 0.0.0.0 --port 8000