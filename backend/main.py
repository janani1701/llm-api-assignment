import os
import uuid
import time
import json
import hashlib
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request, Response
from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy import text
from openai import OpenAI

from db import engine, init_db
import chat_pb2  # generated from protos/chat.proto

load_dotenv()

app = FastAPI()
client = OpenAI()  # reads OPENAI_API_KEY from env

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CLIENT_API_TOKEN = os.getenv("CLIENT_API_TOKEN", "")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))

# super simple in-memory rate limit: {token: [timestamps]}
RATE_BUCKET: Dict[str, list] = {}


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY", "")),
        "client_token_set": bool(CLIENT_API_TOKEN),
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
    }


# ---------------------------
# Auth + Rate limit
# ---------------------------
def require_auth(authorization: Optional[str]):
    # If CLIENT_API_TOKEN not configured, allow (dev mode)
    if not CLIENT_API_TOKEN:
        return "dev"

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if token != CLIENT_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    return token


def rate_limit(token: str):
    now = time.time()
    window_start = now - 60
    timestamps = RATE_BUCKET.get(token, [])
    timestamps = [t for t in timestamps if t >= window_start]

    if len(timestamps) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    timestamps.append(now)
    RATE_BUCKET[token] = timestamps


# ---------------------------
# Idempotency helper
# ---------------------------
def hash_request(conversation_id: str, message: str, model: str) -> str:
    payload = {"conversation_id": conversation_id, "message": message, "model": model}
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------
# Analytics (events) helpers
# ---------------------------
def user_id_from_token(token: str) -> str:
    # don't store raw token; store a short hash
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _emit_event_sync(
    event_name: str,
    token: str,
    conversation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    meta: Optional[dict] = None,
):
    uid = user_id_from_token(token)
    meta_json = json.dumps(meta or {})
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO analytics_events (event_name, user_id, conversation_id, request_id, metadata_json)
                VALUES (:e, :u, :c, :r, :m)
            """),
            {"e": event_name, "u": uid, "c": conversation_id, "r": request_id, "m": meta_json},
        )


def emit_event(
    background_tasks: Optional[BackgroundTasks],
    event_name: str,
    token: str,
    conversation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    meta: Optional[dict] = None,
):
    # async-ish: don't slow down request path
    if background_tasks is not None:
        background_tasks.add_task(_emit_event_sync, event_name, token, conversation_id, request_id, meta)
    else:
        _emit_event_sync(event_name, token, conversation_id, request_id, meta)


# ---------------------------
# API models (JSON)
# ---------------------------
class CreateConversationResponse(BaseModel):
    conversation_id: str


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


# ---------------------------
# Core chat handler (shared by JSON + Protobuf)
# ---------------------------
def handle_chat_core(
    *,
    token: str,
    conversation_id: str,
    message: str,
    background_tasks: Optional[BackgroundTasks],
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    # event: request started
    emit_event(background_tasks, "chat_requested", token, conversation_id=conversation_id, meta={"endpoint": "chat"})

    # ---- Idempotency replay ----
    if idempotency_key:
        req_hash = hash_request(conversation_id, message, MODEL)
        with engine.begin() as conn:
            row = conn.execute(
                text("""
                    SELECT request_hash, response_json
                    FROM idempotency_keys
                    WHERE idempotency_key = :k
                """),
                {"k": idempotency_key},
            ).fetchone()

        if row:
            stored_hash, response_json = row
            if stored_hash != req_hash:
                raise HTTPException(status_code=409, detail="Idempotency key reused with different request")

            emit_event(
                background_tasks,
                "idempotency_replay",
                token,
                conversation_id=conversation_id,
                meta={"idempotency_key": idempotency_key},
            )
            return json.loads(response_json)

    request_id = "r_" + uuid.uuid4().hex
    t0 = time.time()

    # Ensure conversation exists
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT conversation_id FROM conversations WHERE conversation_id = :cid"),
            {"cid": conversation_id},
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="conversation_id not found. Create one first.")

    # Save user message
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:cid, :role, :content)
            """),
            {"cid": conversation_id, "role": "user", "content": message},
        )

    # Build messages[] from DB history (last N)
    N = 12
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content
                FROM messages
                WHERE conversation_id = :cid
                ORDER BY id DESC
                LIMIT :lim
            """),
            {"cid": conversation_id, "lim": N},
        ).fetchall()

    history = [{"role": r[0], "content": r[1]} for r in rows][::-1]
    system_msg = {"role": "system", "content": "You are a helpful assistant. Keep answers clear and concise."}
    messages = [system_msg] + history

    # Call OpenAI
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages)
    except Exception as e:
        emit_event(
            background_tasks,
            "chat_failed",
            token,
            conversation_id=conversation_id,
            request_id=request_id,
            meta={"error": str(e)},
        )
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {str(e)}")

    reply = resp.choices[0].message.content or ""
    latency_ms = int((time.time() - t0) * 1000)

    # Save assistant reply
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:cid, :role, :content)
            """),
            {"cid": conversation_id, "role": "assistant", "content": reply},
        )

    # Usage (may be None)
    prompt_tokens = completion_tokens = total_tokens = None
    if getattr(resp, "usage", None):
        prompt_tokens = resp.usage.prompt_tokens
        completion_tokens = resp.usage.completion_tokens
        total_tokens = resp.usage.total_tokens

    # Insert usage event
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO usage_events (
                    request_id, conversation_id, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    latency_ms
                )
                VALUES (:rid, :cid, :model, :pt, :ct, :tt, :lat)
            """),
            {
                "rid": request_id,
                "cid": conversation_id,
                "model": MODEL,
                "pt": prompt_tokens,
                "ct": completion_tokens,
                "tt": total_tokens,
                "lat": latency_ms,
            },
        )

    response_obj = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "reply": reply,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": MODEL,
            "latency_ms": latency_ms,
        },
    }

    # Store idempotency response (if key provided)
    if idempotency_key:
        req_hash = hash_request(conversation_id, message, MODEL)
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT OR IGNORE INTO idempotency_keys (idempotency_key, request_hash, response_json)
                    VALUES (:k, :h, :r)
                """),
                {"k": idempotency_key, "h": req_hash, "r": json.dumps(response_obj)},
            )

    # event: success
    emit_event(
        background_tasks,
        "chat_succeeded",
        token,
        conversation_id=conversation_id,
        request_id=request_id,
        meta={"model": MODEL, "total_tokens": total_tokens, "latency_ms": latency_ms},
    )

    return response_obj


# ---------------------------
# JSON endpoints
# ---------------------------
@app.post("/v1/conversations", response_model=CreateConversationResponse)
def create_conversation(
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    try:
        token = require_auth(authorization)
    except HTTPException as e:
        emit_event(background_tasks, "auth_failed", "unknown", meta={"endpoint": "/v1/conversations", "status": e.status_code})
        raise

    conversation_id = "c_" + uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO conversations (conversation_id) VALUES (:cid)"),
            {"cid": conversation_id},
        )

    emit_event(background_tasks, "conversation_created", token, conversation_id=conversation_id, meta={"endpoint": "/v1/conversations"})
    return {"conversation_id": conversation_id}


@app.post("/v1/chat/completions")
def chat_json(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    try:
        token = require_auth(authorization)
    except HTTPException as e:
        emit_event(background_tasks, "auth_failed", "unknown", conversation_id=req.conversation_id, meta={"endpoint": "/v1/chat/completions", "status": e.status_code})
        raise

    try:
        rate_limit(token)
    except HTTPException as e:
        if e.status_code == 429:
            emit_event(background_tasks, "rate_limited", token, conversation_id=req.conversation_id, meta={"endpoint": "/v1/chat/completions"})
        raise

    return handle_chat_core(
        token=token,
        conversation_id=req.conversation_id,
        message=req.message,
        background_tasks=background_tasks,
        idempotency_key=idempotency_key,
    )


# ---------------------------
# Protobuf endpoint (FIXED)
# NOTE: your chat_pb2 only has ChatRequest + ChatResponse
# ---------------------------
@app.post("/v1/chat/completions.pb")
async def chat_protobuf(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    # Parse protobuf request body -> ChatRequest
    body = await request.body()
    pb_req = chat_pb2.ChatRequest()
    try:
        pb_req.ParseFromString(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid protobuf payload")

    conversation_id = pb_req.conversation_id
    message = pb_req.message

    # auth + rate limit (same behavior)
    try:
        token = require_auth(authorization)
    except HTTPException as e:
        emit_event(background_tasks, "auth_failed", "unknown", conversation_id=conversation_id, meta={"endpoint": "/v1/chat/completions.pb", "status": e.status_code})
        raise

    try:
        rate_limit(token)
    except HTTPException as e:
        if e.status_code == 429:
            emit_event(background_tasks, "rate_limited", token, conversation_id=conversation_id, meta={"endpoint": "/v1/chat/completions.pb"})
        raise

    # use same core logic
    resp_obj = handle_chat_core(
        token=token,
        conversation_id=conversation_id,
        message=message,
        background_tasks=background_tasks,
        idempotency_key=idempotency_key,
    )

    # Build protobuf response -> ChatResponse (NO usage field here)
    pb_resp = chat_pb2.ChatResponse(
        request_id=resp_obj["request_id"],
        conversation_id=resp_obj["conversation_id"],
        reply=resp_obj["reply"],
    )

    return Response(
        content=pb_resp.SerializeToString(),
        media_type="application/x-protobuf",
    )


# ---------------------------
# Metrics endpoint
# ---------------------------
@app.get("/metrics")
def metrics():
    with engine.begin() as conn:
        dau_rows = conn.execute(text("""
            SELECT date(created_at) AS day, count(distinct user_id) AS dau
            FROM analytics_events
            WHERE event_name='chat_requested'
            GROUP BY day
            ORDER BY day DESC
            LIMIT 30
        """)).fetchall()

        success_rate = conn.execute(text("""
            SELECT
              (SUM(CASE WHEN event_name='chat_succeeded' THEN 1 ELSE 0 END) * 1.0) /
              NULLIF(SUM(CASE WHEN event_name='chat_requested' THEN 1 ELSE 0 END), 0)
            FROM analytics_events
        """)).scalar()

        avg_latency = conn.execute(text("""
            SELECT avg(latency_ms) FROM usage_events
        """)).scalar()

        token_burn = conn.execute(text("""
            SELECT date(created_at) AS day, sum(total_tokens) AS total_tokens
            FROM usage_events
            GROUP BY day
            ORDER BY day DESC
            LIMIT 30
        """)).fetchall()

        # time spent approximation: first/last event per user per day
        time_spent = conn.execute(text("""
            SELECT user_id, date(created_at) AS day,
                   (strftime('%s', max(created_at)) - strftime('%s', min(created_at))) AS seconds_spent
            FROM analytics_events
            GROUP BY user_id, day
            ORDER BY day DESC
            LIMIT 30
        """)).fetchall()

    return {
        "dau_last_30_days": [{"day": r[0], "dau": r[1]} for r in dau_rows],
        "success_rate": float(success_rate) if success_rate is not None else None,
        "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
        "token_burn_last_30_days": [{"day": r[0], "total_tokens": r[1]} for r in token_burn],
        "time_spent_samples": [{"user_id": r[0], "day": r[1], "seconds_spent": r[2]} for r in time_spent],
    }

