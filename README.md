
# LLM API Assignment (FastAPI + SQLite + Usage + Metrics + Protobuf)

This project turns a “web client chatbot” idea into a real backend LLM API that a client (Chrome extension/web UI) can call securely.

## What I built (high level)

A production-style FastAPI backend with clear layers:

1) **Client layer (Chrome extension / web client)**  
Sends chat requests to the backend. It never stores the OpenAI API key.

2) **Auth + Security layer**  
Authorization: Bearer <CLIENT_API_TOKEN> required  
OpenAI key is stored only in backend .env

3) **Protocol layer (clean request/response)**  
POST /v1/conversations → create conversation_id  
POST /v1/chat/completions → chat completion (JSON)  
POST /v1/chat/completions.pb → chat completion (Protobuf optional)

4) **State layer (conversation memory)**  
SQLite tables store conversations + message history  
Backend reconstructs message history and sends it to OpenAI

5) **LLM layer (direct OpenAI API, no frameworks)**  
Calls OpenAI chat completions using server-side SDK

6) **Reliability layer**  
Rate limiting (per token per minute)  
Idempotency (Idempotency-Key) to prevent double-billing / duplicate requests

7) **Telemetry/Analytics layer (usage + product metrics)**  
Stores token usage and latency per request  
Emits analytics events to compute DAU, success rate, time spent, token burn

This matches the requirements:  
Static code analysis friendly (clean structure, DB schema, clear logic)  
Direct OpenAI usage (no LangChain)  
Protocol design: req/res, headers, idempotency, protobuf option

---

## Tech Stack

FastAPI  
SQLite (SQLAlchemy)  
OpenAI API (server-side)  
Protobuf (optional alternative transport)

---

## Project Structure

backend/  
main.py  
db.py  
.env                # ignored by git (secrets)  
.env.example  
app.db              # ignored by git (local sqlite)  
protos/  
chat.proto  
chat_pb2.py         # generated protobuf code  
send_pb.py          # protobuf test client (optional)  

---

## Setup (from scratch)

### 1) Create venv + install dependencies

cd backend  
python -m venv .venv  
source .venv/bin/activate  
pip install -U pip  
pip install fastapi uvicorn python-dotenv sqlalchemy openai protobuf grpcio-tools requests  

### 2) Create .env

Create backend/.env:

OPENAI_API_KEY="PASTE_YOUR_OPENAI_KEY_HERE"  
OPENAI_MODEL="gpt-4o-mini"  
CLIENT_API_TOKEN="lyfg8y4gfb"  
RATE_LIMIT_PER_MIN="30"  

### 3) Run server

cd backend  
source .venv/bin/activate  
uvicorn main:app --reload --port 8080  

Swagger docs:  
http://127.0.0.1:8080/docs  

Health:  
http://127.0.0.1:8080/health  

---

## API usage (JSON)

### Export token

export TOKEN="lyfg8y4gfb"  

### A) Create a conversation

curl -s -X POST "http://127.0.0.1:8080/v1/conversations" \  
  -H "Authorization: Bearer $TOKEN" \  
  -H "Content-Type: application/json" \  
  -d '{}'  

Response:  
{"conversation_id":"c_..."}  

### B) Chat completion

export CID="c_PASTE_ID_HERE"  

curl -s -X POST "http://127.0.0.1:8080/v1/chat/completions" \  
  -H "Authorization: Bearer $TOKEN" \  
  -H "Content-Type: application/json" \  
  -H "Idempotency-Key: idem_test_001" \  
  -d "{\"conversation_id\":\"$CID\",\"message\":\"Tell me a 1 line joke\"}"  

Response includes:  
request_id  
reply  
usage (tokens + latency)  

---

## Idempotency test

### Replay (same key + same request)

Run the same curl again with same Idempotency-Key and same payload.  
Expected: same response replayed (no duplicate OpenAI call).  

### Conflict (same key + different request)

curl -s -X POST "http://127.0.0.1:8080/v1/chat/completions" \  
  -H "Authorization: Bearer $TOKEN" \  
  -H "Content-Type: application/json" \  
  -H "Idempotency-Key: idem_test_001" \  
  -d "{\"conversation_id\":\"$CID\",\"message\":\"Different message\"}"  

Expected:  
HTTP 409  
Idempotency key reused with different request  

---

## Metrics / Analytics

### Metrics endpoint

curl -s http://127.0.0.1:8080/metrics  

Includes:  
DAU (daily active users) last 30 days  
success rate  
avg latency  
token burn last 30 days  
time spent samples (approx using first/last event per user per day)  

---

## Protobuf alternative (Optional)

Why: JSON is verbose and costs more bytes to send. Protobuf is a compact binary protocol.

### 1) Generate protobuf Python file

From backend/:

python -m grpc_tools.protoc -I=protos --python_out=. protos/chat.proto  

### 2) Call protobuf endpoint

Endpoint:  
POST /v1/chat/completions.pb  
Content-Type: application/x-protobuf  

If you have send_pb.py, run:

export TOKEN="lyfg8y4gfb"  
export CID="c_PASTE_ID_HERE"  
python send_pb.py  

---

## Database tables (SQLite)

conversations(conversation_id, created_at)  
messages(id, conversation_id, role, content, created_at)  
usage_events(id, request_id, conversation_id, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, created_at)  
idempotency_keys(id, idempotency_key, request_hash, response_json, created_at)  
analytics_events(id, event_name, user_id, conversation_id, request_id, metadata_json, created_at)  

---

## Notes / Tradeoffs

SQLite chosen for speed + simplicity  
Rate limiting is in-memory (single-process)  
Time spent is approximate (first/last event per user per day)  
OpenAI key is never exposed to client
```

