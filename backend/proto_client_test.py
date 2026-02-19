import os
import requests
import chat_pb2

BASE = "http://127.0.0.1:8080"

TOKEN = os.environ.get("CLIENT_API_TOKEN")
if not TOKEN:
    raise SystemExit("CLIENT_API_TOKEN env var set pannunga")

# 1) Create conversation (JSON)
r = requests.post(
    f"{BASE}/v1/conversations",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json={},
)
r.raise_for_status()
cid = r.json()["conversation_id"]
print("conversation_id:", cid)

# 2) Protobuf chat request
req = chat_pb2.ChatRequest(conversation_id=cid, message="Tell me a 1 line joke")
resp = requests.post(
    f"{BASE}/v1/chat/completions.pb",
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/x-protobuf",
        "Idempotency-Key": "idem_proto_001",
    },
    data=req.SerializeToString(),
)
print("status:", resp.status_code)
resp.raise_for_status()

out = chat_pb2.ChatCompletionResponse()
out.ParseFromString(resp.content)

print("request_id:", out.request_id)
print("reply:", out.reply)
print("total_tokens:", out.usage.total_tokens)
