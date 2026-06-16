const elements = {
  clearButton: document.querySelector("#clear-button"),
  connectionLabel: document.querySelector("#connection-label"),
  connectionPill: document.querySelector("#connection-pill"),
  endpoint: document.querySelector("#endpoint-value"),
  engineSid: document.querySelector("#engine-sid"),
  form: document.querySelector("#message-form"),
  input: document.querySelector("#requirement-input"),
  messages: document.querySelector("#messages"),
  reconnectButton: document.querySelector("#reconnect-button"),
  sendButton: document.querySelector("#send-button"),
  socketSid: document.querySelector("#socket-sid"),
  traceList: document.querySelector("#trace-list"),
};

let socket = null;
let nextAckId = 1;
const pendingAcks = new Map();
let pendingClarification = null;
const resumableStatuses = new Set([
  "awaiting_clarification",
  "awaiting_slot_clarification",
]);

function timestamp() {
  return new Intl.DateTimeFormat("en", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
    hour12: false,
  }).format(new Date());
}

function trace(kind, title, details = "") {
  const entry = document.createElement("div");
  entry.className = "trace-entry";
  entry.dataset.kind = kind;

  const time = document.createElement("span");
  time.className = "trace-time";
  time.textContent = timestamp();

  const type = document.createElement("span");
  type.className = "trace-kind";
  type.textContent = kind;

  const content = document.createElement("span");
  content.className = "trace-content";
  const heading = document.createElement("strong");
  heading.textContent = title;
  content.append(heading);
  if (details) {
    content.append(document.createTextNode(`\n${details}`));
  }

  entry.append(time, type, content);
  elements.traceList.append(entry);
  elements.traceList.scrollTop = elements.traceList.scrollHeight;
}

function setConnectionState(state, label) {
  elements.connectionPill.dataset.state = state;
  elements.connectionLabel.textContent = label;
  elements.sendButton.disabled = state !== "connected";
  trace("state", label, `readyState=${socket?.readyState ?? "none"}`);
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function textBlock(text, className = "subquery-text") {
  const block = document.createElement("pre");
  block.className = className;
  block.textContent = text || "-";
  return block;
}

function appendKnowledgePaths(container, paths = []) {
  if (!paths.length) return;

  const pathList = document.createElement("ul");
  pathList.className = "knowledge-path-list";
  for (const match of paths) {
    const item = document.createElement("li");
    item.textContent = match.path ?? String(match);
    pathList.append(item);
  }
  container.append(pathList);
}

function appendFinalSubqueryPlan(card, plan) {
  const heading = document.createElement("p");
  heading.className = "question-heading";
  heading.textContent = "Final subqueries:";

  const list = document.createElement("div");
  list.className = "subquery-list";

  for (const subquery of plan.final_subqueries ?? []) {
    const section = document.createElement("section");
    section.className = "subquery-card";

    const title = document.createElement("strong");
    title.textContent = `${subquery.slot_type} · ${subquery.channel ?? "Uncategorized"}`;

    const id = document.createElement("code");
    id.textContent = subquery.slot_pack_id;

    const meta = document.createElement("div");
    meta.className = "subquery-meta";
    meta.textContent = subquery.ready_for_generation
      ? "Ready for generation"
      : "Has unresolved placeholders";

    section.append(title, id, meta, textBlock(subquery.final_subquery));

    if (subquery.applied_clarification_answer) {
      const answer = document.createElement("p");
      answer.className = "subquery-note";
      answer.textContent = `Applied answer: ${subquery.applied_clarification_answer}`;
      section.append(answer);
    }

    if (subquery.unresolved_items?.length) {
      const unresolved = document.createElement("ul");
      unresolved.className = "subquery-detail-list";
      for (const itemText of subquery.unresolved_items) {
        const item = document.createElement("li");
        item.textContent = itemText;
        unresolved.append(item);
      }
      section.append(unresolved);
    }

    appendKnowledgePaths(section, subquery.knowledge_paths ?? []);
    list.append(section);
  }

  card.append(heading, list);
}

function addMessage(role, text, result = null) {
  const article = document.createElement("article");
  article.className = `message ${role === "You" ? "user-message" : "agent-message"}`;

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role;

  const card = document.createElement("div");
  card.className = "message-card";
  card.textContent = text;

  if (result?.intent) {
    const intent = result.intent;
    const grid = document.createElement("div");
    grid.className = "result-grid";
    const fields = [
      ["status", result.status],
      ["work product", intent.work_product],
      ["objective", intent.objective],
      ["confidence", intent.confidence],
      ["ready", intent.ready_for_requirement_generation ? "yes" : "no"],
      ["improved query", intent.improved_requirement_query],
    ];
    for (const [name, value] of fields) {
      const key = document.createElement("span");
      key.textContent = name;
      const fieldValue = document.createElement("span");
      fieldValue.textContent = value ?? "-";
      grid.append(key, fieldValue);
    }
    card.append(grid);

    const activeQuestions = result.clarification_questions ?? [];
    const intentQuestions = intent.clarification_questions ?? [];
    const questionsToShow = activeQuestions.length ? activeQuestions : intentQuestions;
    if (questionsToShow.length) {
      const heading = document.createElement("p");
      heading.className = "question-heading";
      heading.textContent = "Please clarify:";
      const questions = document.createElement("ol");
      questions.className = "question-list";
      for (const question of questionsToShow) {
        const item = document.createElement("li");
        item.textContent = question;
        questions.append(item);
      }
      card.append(heading, questions);
    }

    if (intent.relevant_slots?.length) {
      const heading = document.createElement("p");
      heading.className = "question-heading";
      heading.textContent = "Relevant technical slots:";
      const slots = document.createElement("div");
      slots.className = "slot-list";
      for (const slot of intent.relevant_slots) {
        const slotCard = document.createElement("section");
        slotCard.className = "slot-card";
        const title = document.createElement("strong");
        title.textContent = `${slot.feature} · ${slot.slot_type}`;
        const id = document.createElement("code");
        id.textContent = slot.slot_pack_id;
        const themes = document.createElement("ul");
        for (const theme of slot.themes) {
          const item = document.createElement("li");
          item.textContent = theme;
          themes.append(item);
        }
        slotCard.append(title, id, themes);
        slots.append(slotCard);
      }
      card.append(heading, slots);
    }
  }

  if (result?.slot_subquery_plan) {
    const plan = result.slot_subquery_plan;
    const heading = document.createElement("p");
    heading.className = "question-heading";
    heading.textContent = "Slot subquery plan:";
    const summary = document.createElement("pre");
    summary.textContent = pretty({
      requires_clarification: plan.requires_clarification,
      next_clarification_questions: plan.next_clarification_questions,
      subqueries: plan.subqueries?.map((subquery) => ({
        slot_pack_id: subquery.slot_pack_id,
        slot_type: subquery.slot_type,
        ready_for_generation: subquery.ready_for_generation,
      })),
    });
    card.append(heading, summary);
  }

  if (result?.final_subquery_plan) {
    appendFinalSubqueryPlan(card, result.final_subquery_plan);
  }

  article.append(label, card);
  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function sendFrame(frame, description) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    trace("error", "Cannot send frame", "WebSocket is not open.");
    return;
  }
  trace("send", description, frame);
  socket.send(frame);
}

function parseEventFrame(frame) {
  const match = frame.match(/^42(?:\/[^,]+,)?(\d*)(.*)$/s);
  if (!match) return null;
  return {
    ackId: match[1] || null,
    payload: JSON.parse(match[2]),
  };
}

function handleSocketEvent(frame) {
  let parsed;
  try {
    parsed = parseEventFrame(frame);
  } catch (error) {
    trace("error", "Invalid Socket.IO event", String(error));
    return;
  }

  if (!parsed || !Array.isArray(parsed.payload)) return;
  const [eventName, data] = parsed.payload;
  trace("event", eventName, pretty(data));

  if (eventName === "connected") {
    elements.socketSid.textContent = data.sid;
    setConnectionState("connected", "Connected");
    return;
  }

  if (eventName === "requirement_agent:completed") {
    if (data.ok) {
      if (resumableStatuses.has(data.result.status)) {
        pendingClarification = {
          threadId: data.result.thread_id,
          questions: data.result.clarification_questions ?? [],
        };
        elements.input.placeholder =
          data.result.status === "awaiting_slot_clarification"
            ? "Answer the slot clarification questions, then press Enter."
            : "Answer the clarification questions, then press Enter.";
      } else {
        pendingClarification = null;
        elements.input.placeholder = "The system shall report voltage.";
      }
      addMessage("Agent", data.result.intent.intent_summary, data.result);
    } else {
      addMessage("Agent", data.error || "The request failed.");
    }
  }
}

function handleAckFrame(frame) {
  const match = frame.match(/^43(\d+)(.*)$/s);
  if (!match) return;

  const ackId = Number(match[1]);
  let payload = match[2];
  try {
    payload = JSON.parse(payload);
  } catch {
    // Keep the raw payload in the trail if it is not JSON.
  }

  trace("ack", `Acknowledgement #${ackId}`, pretty(payload));
  pendingAcks.delete(ackId);
}

function handleFrame(frame) {
  trace("recv", "Raw frame", frame);

  if (frame.startsWith("0")) {
    const handshake = JSON.parse(frame.slice(1));
    elements.engineSid.textContent = handshake.sid;
    trace("event", "Engine.IO handshake", pretty(handshake));
    sendFrame("40", "Open Socket.IO namespace");
    return;
  }

  if (frame === "2") {
    sendFrame("3", "Engine.IO pong");
    return;
  }

  if (frame.startsWith("42")) {
    handleSocketEvent(frame);
    return;
  }

  if (frame.startsWith("43")) {
    handleAckFrame(frame);
    return;
  }

  if (frame.startsWith("40")) {
    trace("event", "Socket.IO namespace ready", frame.slice(2) || "default namespace");
  }
}

function connect() {
  if (socket) {
    socket.onclose = null;
    socket.close();
  }

  elements.engineSid.textContent = "-";
  elements.socketSid.textContent = "-";
  setConnectionState("connecting", "Connecting");

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const endpoint = `${protocol}//${window.location.host}/socket.io/?EIO=4&transport=websocket`;
  elements.endpoint.textContent = endpoint;
  elements.endpoint.title = endpoint;
  trace("open", "Opening WebSocket", endpoint);

  socket = new WebSocket(endpoint);
  socket.addEventListener("open", () => {
    trace("open", "WebSocket transport open", `protocol=${socket.protocol || "none"}`);
  });
  socket.addEventListener("message", (event) => handleFrame(String(event.data)));
  socket.addEventListener("error", () => {
    trace("error", "WebSocket error", "Check that the FastAPI server is running.");
  });
  socket.addEventListener("close", (event) => {
    setConnectionState("disconnected", "Disconnected");
    trace("close", "WebSocket closed", `code=${event.code} reason=${event.reason || "none"}`);
  });
}

function emitRequirement(requirement) {
  const ackId = nextAckId++;
  const payload = ["requirement_agent:run", { requirement }];
  pendingAcks.set(ackId, { event: payload[0], sentAt: Date.now() });
  sendFrame(`42${ackId}${JSON.stringify(payload)}`, `Emit ${payload[0]} (#${ackId})`);
}

function buildClarificationAnswers(answerText) {
  const questions = pendingClarification?.questions ?? [];
  const lines = answerText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (questions.length && lines.length >= questions.length) {
    return Object.fromEntries(
      questions.map((question, index) => [question, lines[index]]),
    );
  }

  return {
    user_response: answerText,
    questions,
  };
}

function emitClarificationAnswers(answerText) {
  const ackId = nextAckId++;
  const payload = [
    "requirement_agent:run",
    {
      thread_id: pendingClarification.threadId,
      clarification_answers: buildClarificationAnswers(answerText),
    },
  ];
  pendingAcks.set(ackId, { event: payload[0], sentAt: Date.now() });
  sendFrame(`42${ackId}${JSON.stringify(payload)}`, `Resume ${payload[0]} (#${ackId})`);
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const requirement = elements.input.value.trim();
  if (!requirement) return;

  addMessage("You", requirement);
  if (pendingClarification?.threadId) {
    emitClarificationAnswers(requirement);
  } else {
    emitRequirement(requirement);
  }
  elements.input.value = "";
  elements.input.focus();
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

elements.reconnectButton.addEventListener("click", connect);
elements.clearButton.addEventListener("click", () => {
  elements.traceList.replaceChildren();
  trace("state", "Trail cleared", "New transport activity will appear here.");
});

window.addEventListener("beforeunload", () => socket?.close());
connect();
