const baseUrlEl = document.getElementById("baseUrl");
const tokenEl = document.getElementById("token");
const statusEl = document.getElementById("status");
const saveBtn = document.getElementById("save");

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function loadSettings() {
  const data = await chrome.storage.sync.get(["baseUrl", "token"]);
  baseUrlEl.value = data.baseUrl || "http://127.0.0.1:8080";
  tokenEl.value = data.token || "";
}

async function saveSettings() {
  const baseUrl = baseUrlEl.value.trim() || "http://127.0.0.1:8080";
  const token = tokenEl.value.trim();

  await chrome.storage.sync.set({ baseUrl, token });
  setStatus("Saved ✅");
}

saveBtn.addEventListener("click", () => {
  saveSettings().catch((e) => setStatus("Error: " + (e.message || String(e))));
});

loadSettings().catch((e) => setStatus("Error: " + (e.message || String(e))));

