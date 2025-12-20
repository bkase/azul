import init, {
  action_space_size,
  new_game_state,
  observation_size,
} from "./pkg/azul_wasm.js";
import { numpy as np } from "@jax-js/jax";

const byId = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing element: ${id}`);
  }
  return el as T;
};

const ui = {
  status: byId<HTMLSpanElement>("status-text"),
  round: byId<HTMLSpanElement>("round-text"),
  scores: byId<HTMLSpanElement>("scores-text"),
  board: byId<HTMLPreElement>("board-text"),
  turnPill: byId<HTMLDivElement>("turn-pill"),
  checkpointUrl: byId<HTMLInputElement>("checkpoint-url"),
  reloadModel: byId<HTMLButtonElement>("reload-model"),
  newGame: byId<HTMLButtonElement>("new-game"),
  aiFirst: byId<HTMLInputElement>("ai-first"),
  mctsSims: byId<HTMLInputElement>("mcts-sims"),
  actionSelect: byId<HTMLSelectElement>("action-select"),
  playMove: byId<HTMLButtonElement>("play-move"),
  randomMove: byId<HTMLButtonElement>("random-move"),
  log: byId<HTMLDivElement>("log"),
};

type GameHandle = ReturnType<typeof new_game_state>;

type Model = {
  obsSize: number;
  hiddenSize: number;
  predict: (obs: Float32Array | number[]) => Promise<{
    policy: Float32Array;
    value: number;
  }>;
};

let game: GameHandle | null = null;
let model: Model | null = null;
let humanPlayer = 0;
let actionSpace = 0;

const config = {
  cpuct: 1.5,
  maxDepth: 200,
};

function setStatus(text: string) {
  ui.status.textContent = text;
}

function appendLog(line: string) {
  ui.log.textContent += `${line}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function flattenArray(arr: unknown): unknown[] {
  if (!Array.isArray(arr)) {
    return [arr];
  }
  return arr.flat(Infinity);
}

async function readArray(array: any): Promise<Float32Array> {
  if (array == null) {
    return new Float32Array();
  }
  if (typeof array.dataSync === "function") {
    return array.dataSync();
  }
  if (typeof array.toArray === "function") {
    const out = array.toArray();
    const resolved = out instanceof Promise ? await out : out;
    return Float32Array.from(flattenArray(resolved) as number[]);
  }
  if (array.data) {
    return array.data as Float32Array;
  }
  throw new Error("Unable to read array data from jax-js array");
}

function decodeFloat16(uint16: number): number {
  const s = (uint16 & 0x8000) >> 15;
  const e = (uint16 & 0x7c00) >> 10;
  const f = uint16 & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  }
  if (e === 0x1f) {
    return f ? NaN : s ? -Infinity : Infinity;
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}

function float16ToFloat32(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = decodeFloat16(view.getUint16(i * 2, true));
  }
  return out;
}

function makeJaxArray(tensor: any) {
  const data = tensor.data ?? tensor;
  const shape = tensor.shape ?? tensor.s;
  if (!shape) {
    throw new Error("Tensor missing shape metadata");
  }
  return np.array(data).reshape(shape);
}

function parseSafetensors(buffer: ArrayBuffer) {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  const base = 8 + headerLen;

  const tensors: Record<
    string,
    { data: Float32Array | BigInt64Array | Int32Array; shape: number[]; dtype: string }
  > = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [start, end] = (info as { data_offsets: [number, number] }).data_offsets;
    const bytes = new Uint8Array(buffer.slice(base + start, base + end));
    let data: Float32Array | BigInt64Array | Int32Array;
    const dtype = (info as { dtype: string }).dtype;
    if (dtype === "F32") {
      data = new Float32Array(bytes.buffer);
    } else if (dtype === "F16") {
      data = float16ToFloat32(bytes);
    } else if (dtype === "I64") {
      data = new BigInt64Array(bytes.buffer);
    } else if (dtype === "I32") {
      data = new Int32Array(bytes.buffer);
    } else {
      throw new Error(`Unsupported dtype ${dtype} for tensor ${name}`);
    }
    tensors[name] = {
      data,
      shape: (info as { shape: number[] }).shape,
      dtype,
    };
  }
  return tensors;
}

function prepareLinear(
  tensors: Record<string, { data: any; shape: number[]; dtype: string }>,
  name: string,
  inSize: number,
) {
  const t = tensors[name];
  if (!t) {
    throw new Error(`Missing tensor: ${name}`);
  }
  const shape = t.shape;
  if (!shape || shape.length !== 2) {
    throw new Error(`Tensor ${name} expected 2D shape`);
  }
  const w = makeJaxArray(t);
  let outSize;
  let wT;
  if (shape[1] === inSize) {
    outSize = shape[0];
    wT = np.transpose(w);
  } else if (shape[0] === inSize) {
    outSize = shape[1];
    wT = w;
  } else {
    throw new Error(`Tensor ${name} shape ${shape} does not match input ${inSize}`);
  }
  return { wT, outSize };
}

function prepareBias(
  tensors: Record<string, { data: any; shape: number[]; dtype: string }>,
  name: string,
) {
  const t = tensors[name];
  if (!t) {
    return null;
  }
  return makeJaxArray(t);
}

function relu(x: any) {
  return np.maximum(x, 0);
}

function linear(x: any, wT: any, b: any) {
  let y = np.dot(x, wT.ref);
  if (b) {
    y = y.add(b.ref);
  }
  return y;
}

function buildModel(
  tensors: Record<string, { data: any; shape: number[]; dtype: string }>,
  obsSize: number,
): Model {
  const trunk0 = prepareLinear(tensors, "trunk.layers.0.weight", obsSize);
  const trunk0b = prepareBias(tensors, "trunk.layers.0.bias");
  const trunk1 = prepareLinear(tensors, "trunk.layers.2.weight", trunk0.outSize);
  const trunk1b = prepareBias(tensors, "trunk.layers.2.bias");

  const policy0 = prepareLinear(tensors, "policy_head.layers.0.weight", trunk1.outSize);
  const policy0b = prepareBias(tensors, "policy_head.layers.0.bias");
  const policy1 = prepareLinear(tensors, "policy_head.layers.2.weight", policy0.outSize);
  const policy1b = prepareBias(tensors, "policy_head.layers.2.bias");

  const value0 = prepareLinear(tensors, "value_head.layers.0.weight", trunk1.outSize);
  const value0b = prepareBias(tensors, "value_head.layers.0.bias");
  const value1 = prepareLinear(tensors, "value_head.layers.2.weight", value0.outSize);
  const value1b = prepareBias(tensors, "value_head.layers.2.bias");

  return {
    obsSize,
    hiddenSize: trunk0.outSize,
    async predict(obs: Float32Array | number[]) {
      const x = np.array(obs).reshape([1, obsSize]);

      let h = relu(linear(x, trunk0.wT, trunk0b));
      h = relu(linear(h, trunk1.wT, trunk1b));

      let p = relu(linear(h.ref, policy0.wT, policy0b));
      p = linear(p, policy1.wT, policy1b);

      let v = relu(linear(h, value0.wT, value0b));
      v = linear(v, value1.wT, value1b);
      v = np.tanh(v);

      const pFlat = np.reshape(p, [actionSpace]);
      const vFlat = np.reshape(v, [1]);
      const policyArray = await readArray(pFlat.ref);
      const valueArray = await readArray(vFlat.ref);

      pFlat.dispose();
      vFlat.dispose();
      return {
        policy: policyArray,
        value: valueArray[0] ?? valueArray,
      };
    },
  };
}

async function loadModel() {
  setStatus("Loading model…");
  const url = ui.checkpointUrl.value.trim();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch checkpoint: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  const tensors = parseSafetensors(buffer);
  actionSpace = action_space_size();
  const obsSize = observation_size();
  const modelObj = buildModel(tensors, obsSize);

  const policyShape = (tensors["policy_head.layers.2.weight"].shape ?? [])[0];
  if (policyShape && policyShape !== actionSpace) {
    console.warn("Policy head size mismatch", policyShape, actionSpace);
  }
  model = modelObj;
  setStatus("Model ready");
}

function softmaxLegal(logits: Float32Array, legalIds: Uint16Array | number[]) {
  let maxLogit = -Infinity;
  for (const id of legalIds) {
    const v = logits[id];
    if (v > maxLogit) maxLogit = v;
  }
  let sum = 0;
  const result: Array<[number, number]> = [];
  for (const id of legalIds) {
    const value = Math.exp(logits[id] - maxLogit);
    sum += value;
    result.push([id, value]);
  }
  if (sum <= 0) {
    return result.map(([id]) => [id, 1 / legalIds.length]);
  }
  return result.map(([id, value]) => [id, value / sum]);
}

function selectChild(node: MctsNode, cpuct: number) {
  let bestIdx = 0;
  let bestScore = -Infinity;
  const parentN = Math.max(1, node.visitCount);
  node.children.forEach((edge, idx) => {
    const q = edge.visitCount > 0 ? edge.valueSum / edge.visitCount : 0;
    const u = cpuct * edge.prior * (Math.sqrt(parentN) / (1 + edge.visitCount));
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      bestIdx = idx;
    }
  });
  return bestIdx;
}

class MctsNode {
  state: GameHandle;
  toPlay: number;
  isTerminal: boolean;
  children: Array<{
    actionId: number;
    prior: number;
    visitCount: number;
    valueSum: number;
    reward: number;
    child: MctsNode | null;
  }>;
  visitCount: number;

  constructor(state: GameHandle) {
    this.state = state;
    this.toPlay = state.current_player();
    this.isTerminal = state.is_game_over();
    this.children = [];
    this.visitCount = 0;
  }
}

async function evaluateNode(node: MctsNode) {
  if (!model) {
    throw new Error("Model not loaded");
  }
  if (node.isTerminal) {
    return 0;
  }
  if (node.children.length > 0) {
    return 0;
  }
  const obs = node.state.encode_observation(node.toPlay);
  const { policy, value } = await model.predict(obs);
  const legalIds = node.state.legal_action_ids();
  const priors = softmaxLegal(policy, legalIds);
  node.children = priors.map(([id, prior]) => ({
    actionId: id,
    prior,
    visitCount: 0,
    valueSum: 0,
    reward: 0,
    child: null,
  }));
  return value;
}

async function runSimulation(root: MctsNode) {
  const path: Array<{ node: MctsNode; edge: MctsNode["children"][number] }> = [];
  let node = root;
  for (let depth = 0; depth < config.maxDepth; depth += 1) {
    if (node.isTerminal || node.children.length === 0) {
      break;
    }
    const idx = selectChild(node, config.cpuct);
    const edge = node.children[idx];
    path.push({ node, edge });

    if (!edge.child) {
      const childState = node.state.clone_handle();
      const applyResult = childState.apply_action_id(edge.actionId);
      edge.reward = applyResult.reward as number;
      edge.child = new MctsNode(childState);
      node = edge.child;
      break;
    }
    node = edge.child;
  }

  let leafValue = 0;
  if (!node.isTerminal) {
    leafValue = await evaluateNode(node);
  }

  let value = leafValue;
  for (let i = path.length - 1; i >= 0; i -= 1) {
    const { node: parent, edge } = path[i];
    const child = edge.child as MctsNode;
    if (parent.toPlay !== child.toPlay) {
      value = -value;
    }
    value = Math.max(-1, Math.min(1, edge.reward + value));
    edge.visitCount += 1;
    edge.valueSum += value;
    parent.visitCount += 1;
  }
}

async function selectAction(rootState: GameHandle, numSimulations: number) {
  const root = new MctsNode(rootState);
  await evaluateNode(root);
  for (let i = 0; i < numSimulations; i += 1) {
    await runSimulation(root);
  }
  let bestId = root.children[0]?.actionId ?? 0;
  let bestCount = -Infinity;
  for (const edge of root.children) {
    if (edge.visitCount > bestCount) {
      bestCount = edge.visitCount;
      bestId = edge.actionId;
    }
  }
  return bestId;
}

function updateActionList() {
  if (!game) return;
  const ids = game.legal_action_ids();
  const labels = game.legal_action_strings();
  ui.actionSelect.innerHTML = "";
  ids.forEach((id, idx) => {
    const option = document.createElement("option");
    const label = labels[idx] ?? `Action ${id}`;
    option.value = String(id);
    option.textContent = `${idx}: ${label}`;
    ui.actionSelect.appendChild(option);
  });
  ui.playMove.disabled = ids.length === 0;
  ui.randomMove.disabled = ids.length === 0;
}

function updateBoard() {
  if (!game) return;
  ui.board.textContent = game.render_text(humanPlayer);
  ui.turnPill.textContent =
    game.current_player() === humanPlayer ? "Your turn" : "AI thinking";
  ui.round.textContent = `${game.round() + 1}`;
  const scores = Array.from(game.scores());
  ui.scores.textContent = scores.map((s, idx) => `P${idx}: ${s}`).join(" · ");
  updateActionList();
}

function setControlsEnabled(enabled: boolean) {
  ui.playMove.disabled = !enabled;
  ui.randomMove.disabled = !enabled;
  ui.actionSelect.disabled = !enabled;
}

async function maybeRunAi() {
  if (!game) return;
  while (!game.is_game_over() && game.current_player() !== humanPlayer) {
    setStatus("AI thinking…");
    ui.turnPill.textContent = "AI thinking";
    setControlsEnabled(false);
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const sims = Number.parseInt(ui.mctsSims.value, 10) || 400;
    const rootState = game.clone_handle();
    const actionId = await selectAction(rootState, sims);
    const actionLabel = game.action_id_to_string(actionId);
    const result = game.apply_action_id(actionId);
    appendLog(`AI: ${actionLabel}`);
    updateBoard();
    if (result.game_over) {
      announceGameOver(result);
      return;
    }
  }
  setStatus("Your turn");
  setControlsEnabled(true);
}

function announceGameOver(result: any) {
  const scores = result.scores ?? [];
  const humanScore = scores[humanPlayer];
  const aiScore = scores[1 - humanPlayer];
  setStatus("Game over");
  ui.turnPill.textContent = "Game over";
  appendLog(`Game over. You ${humanScore >= aiScore ? "win" : "lose"}!`);
}

async function handleHumanMove(actionId: number) {
  if (!game) return;
  const label = game.action_id_to_string(actionId);
  const result = game.apply_action_id(actionId);
  appendLog(`You: ${label}`);
  updateBoard();
  if (result.game_over) {
    announceGameOver(result);
    return;
  }
  await maybeRunAi();
}

function newGame() {
  const seed = BigInt(Date.now());
  game = new_game_state(seed);
  humanPlayer = ui.aiFirst.checked ? 1 : 0;
  ui.log.textContent = "";
  updateBoard();
  setStatus("Your turn");
  if (game.current_player() !== humanPlayer) {
    void maybeRunAi();
  }
}

ui.playMove.addEventListener("click", () => {
  const actionId = Number.parseInt(ui.actionSelect.value, 10);
  if (!Number.isNaN(actionId)) {
    void handleHumanMove(actionId);
  }
});

ui.randomMove.addEventListener("click", () => {
  if (!game) return;
  const ids = game.legal_action_ids();
  if (ids.length === 0) return;
  const randomId = ids[Math.floor(Math.random() * ids.length)];
  void handleHumanMove(randomId);
});

ui.newGame.addEventListener("click", () => newGame());
ui.reloadModel.addEventListener("click", async () => {
  try {
    await loadModel();
  } catch (err: any) {
    console.error(err);
    setStatus(`Model load failed: ${err?.message ?? err}`);
  }
});

async function boot() {
  setStatus("Loading WASM…");
  await init();
  try {
    await loadModel();
  } catch (err: any) {
    console.error(err);
    setStatus(`Model load failed: ${err?.message ?? err}`);
    return;
  }
  newGame();
}

void boot();
