const msgEl = document.getElementById("msg");
const outEl = document.getElementById("out");
const sendBtn = document.getElementById("send");
const openOptionsEl = document.getElementById("openOptions");

function setOut(text) {
  outEl.textContent = text;
}

async function getSettings() {
  const data = await chrome.storage.sync.get(["baseUrl", "token", "conversationId"]);
  return {
    baseUrl: data.baseUrl || "http://127.0.0.1:8080",
    token: data.token || "",
    conversationId: data.conversationId || null
  };
}

async function setConversationId(conversationId) {
  await chrome.storage.sync.set({ conversationId });
}

openOptionsEl.addEventListener("click", async (e) => {
  e.preventDefault();
  if (chrome.runtime.openOptionsPage) chrome.runtime.openOptionsPage();
});

async function createConversation(baseUrl, token) {
  const res = await fetch(`${baseUrl}/v1/conversations`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({})
  });

  const json = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(json));
  return json.conversation_id;
}

async function sendChat(baseUrl, token, conversationId, message) {
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json",
      "Idempotency-Key": `ext_${Date.now()}`
    },
    body: JSON.stringify({
      conversation_id: conversationId,
      message
    })
  });

  const json = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(json));
  return json;
}

sendBtn.addEventListener("click", async () => {
  try {
    const message = msgEl.value.trim();
    if (!message) return setOut("Message missing");

    setOut("Loading settings...");

    const { baseUrl, token, conversationId } = await getSettings();
    if (!token) return setOut("Client token missing. Go to Settings and paste CLIENT_API_TOKEN.");

    let cid = conversationId;
    if (!cid) {
      setOut("Creating conversation...");
      cid = await createConversation(baseUrl, token);
      await setConversationId(cid);
    }

    setOut("Calling chat...");
    const result = await sendChat(baseUrl, token, cid, message);

    setOut(result.reply || JSON.stringify(result, null, 2));
  } catch (e) {
    setOut("Error: " + (e.message || String(e)));
  }
});

