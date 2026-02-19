import os
import requests
import chat_pb2

TOKEN = os.getenv("TOKEN", "")
URL = "http://127.0.0.1:8080/v1/chat/completions.pb"

req = chat_pb2.ChatRequest(
    conversation_id=os.getenv("CID", ""),
    message="Tell me a 1 line joke"
)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Idempotency-Key": "idem_pb_001",
    "Content-Type": "application/x-protobuf",
}

r = requests.post(URL, data=req.SerializeToString(), headers=headers)
print("status:", r.status_code)

if r.status_code != 200:
    print(r.text)
    raise SystemExit(1)

resp = chat_pb2.ChatResponse()
resp.ParseFromString(r.content)

print("request_id:", resp.request_id)
print("conversation_id:", resp.conversation_id)
print("reply:", resp.reply)
