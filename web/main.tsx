import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  RotateCcw,
  User,
  Cpu,
  Binary,
  Zap,
  Target,
  Activity,
  SlidersHorizontal,
} from "lucide-react";
import init, {
  action_space_size,
  new_game_state,
  observation_size,
} from "./pkg/azul_wasm.js";
import { numpy as np } from "@jax-js/jax";

/**
 * BRAND DESIGN TOKENS: Azul AlphaZero (Concise Professional)
 */
const PALETTE = {
  bg: "#F8FAFC",
  panel: "#FFFFFF",
  blue: "#2563EB",
  amber: "#F59E0B",
  emerald: "#10B981",
  rose: "#F43F5E",
  zinc: "#18181B",
  slate: "#64748B",
};

const ELEMENT_TYPES = ["blue", "amber", "emerald", "rose", "zinc"] as const;
const SOURCE_COUNT = 5;
const LATTICE_MAP = [
  ["blue", "amber", "rose", "zinc", "emerald"],
  ["emerald", "blue", "amber", "rose", "zinc"],
  ["zinc", "emerald", "blue", "amber", "rose"],
  ["rose", "zinc", "emerald", "blue", "amber"],
  ["amber", "rose", "zinc", "emerald", "blue"],
] as const;

const LOSS_FACTORS = [-1, -1, -2, -2, -2, -3, -3] as const;
const MAX_ROUNDS = 5;
const DEFAULT_CHECKPOINT_URL = "../checkpoints6/best.safetensors";
const MIN_THINKING_MS = 350;

const byNumber = (value: number | null | undefined, digits = 2) =>
  value == null || Number.isNaN(value) ? "--" : value.toFixed(digits);

const formatMs = (value: number) =>
  value <= 0 || Number.isNaN(value) ? "--" : `${value.toFixed(1)} ms`;

const HISTORY_LEN = 50;

const pushHistory = (arr: number[], value: number, max = HISTORY_LEN) => {
  arr.push(value);
  if (arr.length > max) {
    arr.shift();
  }
};

const mean = (arr: number[]) =>
  arr.length === 0 ? 0 : arr.reduce((sum, v) => sum + v, 0) / arr.length;

const stddev = (arr: number[]) => {
  if (arr.length < 2) return 0;
  const avg = mean(arr);
  const variance =
    arr.reduce((sum, v) => sum + (v - avg) * (v - avg), 0) / (arr.length - 1);
  return Math.sqrt(variance);
};

const percentile = (arr: number[], p: number) => {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor(p * sorted.length)),
  );
  return sorted[idx];
};

const entropyFromProbs = (probs: number[]) => {
  let entropy = 0;
  for (const p of probs) {
    if (p > 0) entropy -= p * Math.log(p);
  }
  return entropy;
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

type ActionSourceView =
  | { kind: "factory"; index: number }
  | { kind: "center" };

type ActionDestView = { kind: "pattern"; row: number } | { kind: "floor" };

type ActionDetail = {
  id: number;
  source: ActionSourceView;
  color: string;
  dest: ActionDestView;
};

type PatternLineView = {
  color: string | null;
  count: number;
  capacity: number;
};

type PlayerView = {
  pattern_lines: PatternLineView[];
  wall: Array<Array<string | null>>;
  floor: string[];
  score: number;
};

type GameStateView = {
  num_players: number;
  current_player: number;
  round: number;
  final_round_triggered: boolean;
  factories: string[][];
  center: string[];
  has_origin: boolean;
  players: PlayerView[];
};

type StatsSnapshot = {
  backend: string;
  inferenceLast: number;
  inferenceAvg: number;
  inferenceP95: number;
  mctsLast: number;
  simsPerSec: number;
  sims: number;
  nodes: number;
  depth: number;
  evals: number;
  valueAvg: number;
  valueStd: number;
  policyEntropy: number;
  mctsEntropy: number;
  convergence: number;
  topMoves: string[];
};

type SourceSelection = { kind: "factory"; index: number } | { kind: "center" };

type SelectionState = {
  source: SourceSelection | null;
  color: string | null;
};

const emptyStats: StatsSnapshot = {
  backend: "--",
  inferenceLast: 0,
  inferenceAvg: 0,
  inferenceP95: 0,
  mctsLast: 0,
  simsPerSec: 0,
  sims: 0,
  nodes: 0,
  depth: 0,
  evals: 0,
  valueAvg: 0,
  valueStd: 0,
  policyEntropy: 0,
  mctsEntropy: 0,
  convergence: 0,
  topMoves: [],
};

const config = {
  cpuct: 1.5,
  maxDepth: 200,
};

const sizeClasses = {
  xs: "w-2.5 h-2.5 rounded-sm",
  sm: "w-4 h-4 rounded-md",
  md: "w-6 h-6 rounded-lg",
  lg: "w-8 h-8 rounded-xl",
};

type NodeProps = {
  type: string;
  size?: keyof typeof sizeClasses;
  onClick?: React.MouseEventHandler<HTMLDivElement>;
  disabled?: boolean;
  selected?: boolean;
  preview?: boolean;
  dataRole?: string;
  dataColor?: string;
  dataIndex?: number;
  ariaLabel?: string;
};

const Node = ({
  type,
  size = "md",
  onClick,
  disabled,
  selected,
  preview,
  dataRole,
  dataColor,
  dataIndex,
  ariaLabel,
}: NodeProps) => {
  const styles: Record<string, string> = {
    blue: "bg-blue-600 border-blue-800",
    amber: "bg-amber-500 border-amber-700",
    emerald: "bg-emerald-500 border-emerald-700",
    rose: "bg-rose-500 border-rose-700",
    zinc: "bg-zinc-800 border-black",
    origin:
      "bg-white border-slate-300 text-slate-400 text-[9px] flex items-center justify-center font-bold",
  };

  return (
    <div
      onClick={!disabled ? onClick : undefined}
      data-role={dataRole}
      data-color={dataColor}
      data-index={typeof dataIndex === "number" ? dataIndex : undefined}
      aria-label={ariaLabel}
      className={`
        ${sizeClasses[size]} ${styles[type] || "bg-slate-200"}
        border-b-[2px] active:border-b-0 active:translate-y-0.5 transition-all cursor-pointer
        relative flex items-center justify-center
        ${disabled ? "opacity-40 cursor-not-allowed scale-95" : "hover:scale-110"}
        ${selected ? "ring-2 ring-blue-500 ring-offset-1 z-10 scale-110 shadow-md" : ""}
        ${preview ? "opacity-15 animate-pulse border-b-0 translate-y-0.5" : ""}
      `}
    >
      {type === "origin" && "1"}
      <div className="absolute inset-0 bg-white/10 pointer-events-none rounded-inherit" />
    </div>
  );
};

class StatsTracker {
  inferenceTimes: number[] = [];
  mctsTimes: number[] = [];
  valueHistory: number[] = [];
  policyEntropyHistory: number[] = [];
  mctsEntropyHistory: number[] = [];
  currentMove: {
    sims: number;
    nodesExpanded: number;
    maxDepth: number;
    evals: number;
    start: number;
  } | null = null;
  snapshot: StatsSnapshot;

  constructor(private onUpdate: (snapshot: StatsSnapshot) => void) {
    const webgpu = typeof navigator !== "undefined" && "gpu" in navigator;
    this.snapshot = {
      ...emptyStats,
      backend: webgpu ? "WebGPU" : "CPU",
    };
    this.onUpdate(this.snapshot);
  }

  reset() {
    this.inferenceTimes = [];
    this.mctsTimes = [];
    this.valueHistory = [];
    this.policyEntropyHistory = [];
    this.mctsEntropyHistory = [];
    this.currentMove = null;
    this.commit({
      mctsLast: 0,
      simsPerSec: 0,
      sims: 0,
      nodes: 0,
      depth: 0,
      evals: 0,
      convergence: 0,
      topMoves: [],
    });
  }

  recordInference(ms: number) {
    pushHistory(this.inferenceTimes, ms);
    this.commit({});
  }

  recordValue(value: number) {
    pushHistory(this.valueHistory, value);
    this.commit({});
  }

  recordPolicyEntropy(entropy: number) {
    pushHistory(this.policyEntropyHistory, entropy);
    this.commit({});
  }

  beginMove(sims: number) {
    this.currentMove = {
      sims,
      nodesExpanded: 0,
      maxDepth: 0,
      evals: 0,
      start: performance.now(),
    };
  }

  recordEval() {
    if (this.currentMove) {
      this.currentMove.evals += 1;
    }
  }

  recordExpansion(depth: number) {
    if (this.currentMove) {
      this.currentMove.nodesExpanded += 1;
      this.currentMove.maxDepth = Math.max(this.currentMove.maxDepth, depth);
    }
  }

  finishMove(root: MctsNode) {
    if (!this.currentMove) return;
    const elapsed = performance.now() - this.currentMove.start;
    pushHistory(this.mctsTimes, elapsed);

    const visitTotal = root.children.reduce(
      (sum, edge) => sum + edge.visitCount,
      0,
    );
    const visitProbs =
      visitTotal > 0
        ? root.children.map((edge) => edge.visitCount / visitTotal)
        : root.children.map(() => 0);
    const mctsEntropy = entropyFromProbs(visitProbs);
    pushHistory(this.mctsEntropyHistory, mctsEntropy);

    const topMoves = root.children
      .slice()
      .sort((a, b) => b.visitCount - a.visitCount)
      .slice(0, 5)
      .map((edge) => {
        const prob = visitTotal > 0 ? edge.visitCount / visitTotal : 0;
        return `${root.state.action_id_to_string(edge.actionId)} (${prob.toFixed(2)})`;
      });

    const maxProb = visitProbs.length ? Math.max(...visitProbs) : 0;
    const simsPerSec = elapsed > 0 ? Math.round((this.currentMove.sims / elapsed) * 1000) : 0;

    this.commit({
      mctsLast: elapsed,
      simsPerSec,
      sims: this.currentMove.sims,
      nodes: this.currentMove.nodesExpanded,
      depth: this.currentMove.maxDepth,
      evals: this.currentMove.evals,
      convergence: maxProb,
      topMoves,
    });

    this.currentMove = null;
  }

  private commit(partial: Partial<StatsSnapshot>) {
    const inferenceLast = this.inferenceTimes.at(-1) ?? 0;
    const inferenceAvg = mean(this.inferenceTimes);
    const inferenceP95 = percentile(this.inferenceTimes, 0.95);
    const valueAvg = mean(this.valueHistory);
    const valueStd = stddev(this.valueHistory);
    const policyEntropy = mean(this.policyEntropyHistory);
    const mctsEntropy = mean(this.mctsEntropyHistory);

    this.snapshot = {
      ...this.snapshot,
      inferenceLast,
      inferenceAvg,
      inferenceP95,
      valueAvg,
      valueStd,
      policyEntropy,
      mctsEntropy,
      ...partial,
    };
    this.onUpdate(this.snapshot);
  }
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
    return f ? Number.NaN : s ? -Infinity : Infinity;
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
  actionSpace: number,
  stats: StatsTracker,
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
      const t0 = performance.now();
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
      stats.recordInference(performance.now() - t0);
      return {
        policy: policyArray,
        value: valueArray[0] ?? valueArray,
      };
    },
  };
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
  nnValue: number | null;
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
    this.nnValue = null;
    this.children = [];
    this.visitCount = 0;
  }
}

async function evaluateNode(node: MctsNode, model: Model, stats: StatsTracker) {
  if (node.isTerminal) {
    return 0;
  }
  if (node.children.length > 0) {
    return 0;
  }
  const obs = node.state.encode_observation(node.toPlay);
  const { policy, value } = await model.predict(obs);
  stats.recordEval();
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
  node.nnValue = value;
  stats.recordValue(value);
  stats.recordPolicyEntropy(entropyFromProbs(priors.map(([, prob]) => prob)));
  return value;
}

async function runSimulation(
  root: MctsNode,
  model: Model,
  stats: StatsTracker,
) {
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
      stats.recordExpansion(depth + 1);
      node = edge.child;
      break;
    }
    node = edge.child;
  }

  let leafValue = 0;
  if (!node.isTerminal) {
    leafValue = await evaluateNode(node, model, stats);
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

async function selectAction(
  rootState: GameHandle,
  numSimulations: number,
  model: Model,
  stats: StatsTracker,
) {
  const root = new MctsNode(rootState);
  stats.beginMove(numSimulations);
  await evaluateNode(root, model, stats);
  for (let i = 0; i < numSimulations; i += 1) {
    await runSimulation(root, model, stats);
  }
  stats.finishMove(root);

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

const App = () => {
  const [status, setStatus] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [stateView, setStateView] = useState<GameStateView | null>(null);
  const [legalActions, setLegalActions] = useState<ActionDetail[]>([]);
  const [selection, setSelection] = useState<SelectionState>({
    source: null,
    color: null,
  });
  const [confirmBufferIdx, setConfirmBufferIdx] = useState<number | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [mctsSims, setMctsSims] = useState(400);
  const [statsSnapshot, setStatsSnapshot] = useState<StatsSnapshot>(emptyStats);
  const [modelReady, setModelReady] = useState(false);
  const [aiThinking, setAiThinking] = useState(false);

  const gameRef = useRef<GameHandle | null>(null);
  const modelRef = useRef<Model | null>(null);
  const statsRef = useRef<StatsTracker | null>(null);
  const mctsSimsRef = useRef(mctsSims);
  const aiThinkingRef = useRef(false);

  const humanPlayer = 0;
  const aiPlayer = 1;

  const appendLog = useCallback((line: string) => {
    setLogLines((prev) => {
      const next = [...prev, line];
      return next.slice(-100);
    });
  }, []);

  const refreshView = useCallback(() => {
    const game = gameRef.current;
    if (!game) return;
    try {
      const viewRaw = game.state_view() as unknown as GameStateView;
      const view: GameStateView = {
        ...viewRaw,
        current_player: Number(viewRaw.current_player),
        round: Number(viewRaw.round),
        num_players: Number(viewRaw.num_players),
      };
      const actions = game.legal_action_details() as unknown as ActionDetail[];
      setStateView(view);
      setLegalActions(actions);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const initialize = useCallback(() => {
    const stats = statsRef.current;
    if (stats) {
      stats.reset();
    }
    const seed = BigInt(Math.floor(Math.random() * 2 ** 32));
    gameRef.current = new_game_state(seed);
    setSelection({ source: null, color: null });
    setConfirmBufferIdx(null);
    setLogLines([]);
    refreshView();
  }, [refreshView]);

  useEffect(() => {
    statsRef.current = new StatsTracker(setStatsSnapshot);
  }, []);

  useEffect(() => {
    mctsSimsRef.current = mctsSims;
  }, [mctsSims]);

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      try {
        setStatus("Loading engine...");
        const wasmUrl = new URL("./pkg/azul_wasm_bg.wasm", import.meta.url);
        wasmUrl.search = `v=${Date.now()}`;
        await init(wasmUrl);
        if (cancelled) return;

        setStatus("Loading model...");
        const response = await fetch(DEFAULT_CHECKPOINT_URL);
        if (!response.ok) {
          throw new Error(`Failed to fetch checkpoint: ${response.status}`);
        }
        const buffer = await response.arrayBuffer();
        const tensors = parseSafetensors(buffer);
        const actionSpace = action_space_size();
        const obsSize = observation_size();
        const stats = statsRef.current;
        if (!stats) {
          throw new Error("Stats tracker unavailable");
        }
        const modelObj = buildModel(tensors, obsSize, actionSpace, stats);

        const policyShape = (tensors["policy_head.layers.2.weight"].shape ?? [])[0];
        if (policyShape && policyShape !== actionSpace) {
          console.warn("Policy head size mismatch", policyShape, actionSpace);
        }

        if (cancelled) return;
        modelRef.current = modelObj;
        setModelReady(true);
        setStatus("Model ready");
        initialize();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setStatus("Failed to load");
      }
    };

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, [initialize]);

  const runAiTurn = useCallback(async () => {
    if (aiThinkingRef.current) return;
    const game = gameRef.current;
    const model = modelRef.current;
    const stats = statsRef.current;
    if (!game || !model || !stats) return;

    aiThinkingRef.current = true;
    setAiThinking(true);
    setStatus("AlphaZero thinking...");
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const thinkingStart = performance.now();

    try {
      while (!game.is_game_over() && game.current_player() === aiPlayer) {
        const sims = mctsSimsRef.current;
        const rootState = game.clone_handle();
        const actionId = await selectAction(rootState, sims, model, stats);
        const label = game.action_id_to_string(actionId);
        game.apply_action_id(actionId);
        appendLog(`AI: ${label}`);
        setSelection({ source: null, color: null });
        setConfirmBufferIdx(null);
        refreshView();
      }
    } finally {
      const elapsed = performance.now() - thinkingStart;
      if (elapsed < MIN_THINKING_MS) {
        await new Promise((resolve) => setTimeout(resolve, MIN_THINKING_MS - elapsed));
      }
      aiThinkingRef.current = false;
      setAiThinking(false);
      setStatus(game.is_game_over() ? "Game over" : "Your move");
    }
  }, [appendLog, refreshView]);

  useEffect(() => {
    if (!stateView || !modelReady) return;
    const game = gameRef.current;
    if (!game || game.is_game_over()) return;
    if (stateView.current_player === aiPlayer) {
      void runAiTurn();
    } else {
      setStatus("Your move");
    }
  }, [stateView, modelReady, runAiTurn]);

  const handleSourceSelection = useCallback(
    (source: SourceSelection, color: string | null = null) => {
      if (!stateView || stateView.current_player !== humanPlayer) return;
      setSelection({ source, color });
      setConfirmBufferIdx(null);
    },
    [stateView],
  );

  const legalForSelection = useMemo(() => {
    const source = selection.source;
    const color = selection.color;
    if (!source || !color) return [];
    return legalActions.filter((action) => {
      if (action.color !== color) return false;
      if (source.kind === "factory") {
        return action.source.kind === "factory" && action.source.index === source.index;
      }
      return action.source.kind === "center";
    });
  }, [legalActions, selection.color, selection.source]);

  const legalRows = useMemo(() => {
    const rows = new Set<number>();
    for (const action of legalForSelection) {
      if (action.dest.kind === "pattern") {
        rows.add(action.dest.row);
      }
    }
    return rows;
  }, [legalForSelection]);

  const legalFloor = useMemo(
    () => legalForSelection.some((action) => action.dest.kind === "floor"),
    [legalForSelection],
  );

  const selectionCount = useMemo(() => {
    if (!stateView || !selection.source || !selection.color) return 0;
    if (selection.source.kind === "factory") {
      const factory = stateView.factories[selection.source.index] ?? [];
      return factory.filter((t) => t === selection.color).length;
    }
    return stateView.center.filter((t) => t === selection.color).length;
  }, [selection, stateView]);

  const selectionTakesOrigin =
    selection.source?.kind === "center" && !!stateView?.has_origin;

  const handleBufferInteraction = useCallback(
    (bufferIdx: number) => {
      if (!stateView || !selection.source || !selection.color) return;
      if (stateView.current_player !== humanPlayer) return;
      if (bufferIdx === -1) {
        if (!legalFloor) return;
      } else if (!legalRows.has(bufferIdx)) {
        return;
      }

      if (confirmBufferIdx !== bufferIdx) {
        setConfirmBufferIdx(bufferIdx);
        return;
      }

      const action = legalForSelection.find((candidate) => {
        if (bufferIdx === -1) {
          return candidate.dest.kind === "floor";
        }
        return candidate.dest.kind === "pattern" && candidate.dest.row === bufferIdx;
      });

      if (!action) return;

      const game = gameRef.current;
      if (!game) return;
      const label = game.action_id_to_string(action.id);
      game.apply_action_id(action.id);
      appendLog(`You: ${label}`);
      setSelection({ source: null, color: null });
      setConfirmBufferIdx(null);
      refreshView();
    },
    [
      appendLog,
      confirmBufferIdx,
      legalFloor,
      legalForSelection,
      legalRows,
      refreshView,
      selection,
      stateView,
    ],
  );

  const gameOver = gameRef.current?.is_game_over() ?? false;
  const aiTurn = stateView?.current_player === aiPlayer && !gameOver;
  const isHumanTurn = stateView?.current_player === humanPlayer && !aiThinking;

  const agents = useMemo(() => {
    if (!stateView) return [];
    return stateView.players.map((player, idx) => {
      const buffers = player.pattern_lines.map((line) => {
        const tiles = Array.from({ length: line.capacity }, (_, i) =>
          i < line.count && line.color ? line.color : null,
        );
        return tiles;
      });
      return {
        id: idx,
        name: idx === humanPlayer ? "You" : "AlphaZero",
        buffers,
        lattice: player.wall,
        inertia: player.floor,
        score: player.score,
      };
    });
  }, [stateView]);

  const statusDetail = error
    ? `Error: ${error}`
    : gameOver
      ? "Game over"
      : aiThinking || aiTurn
        ? "AI thinking..."
        : status;

  return (
    <div
      className="h-screen text-slate-900 font-display flex flex-col overflow-hidden"
      style={{
        background:
          "radial-gradient(circle at top, rgba(248,250,252,1) 0%, rgba(241,245,249,1) 42%, rgba(226,232,240,1) 100%)",
      }}
    >
      <header className="bg-white/90 backdrop-blur border-b border-slate-200 px-3 h-10 flex items-center justify-between shrink-0 z-20">
        <div className="flex items-center gap-2">
          <Binary size={14} className="text-zinc-900" />
          <h1 className="font-black text-sm tracking-tight">Azul AlphaZero</h1>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-[9px] font-bold text-slate-500 uppercase tracking-tighter">
            {statusDetail}
          </div>
          <div className="flex gap-3 text-[9px] font-bold text-slate-500 uppercase tracking-tighter">
            <span>
              Stab:{" "}
              <span className="text-blue-600">
                {statsSnapshot.valueStd > 0
                  ? `${byNumber(statsSnapshot.valueAvg, 2)} +/- ${byNumber(
                      statsSnapshot.valueStd,
                      2,
                    )}`
                  : "--"}
              </span>
            </span>
            <span>
              Rnd:{" "}
              <span className="text-zinc-900">
                {stateView ? `${stateView.round + 1}/${MAX_ROUNDS}` : "--"}
              </span>
            </span>
          </div>
          <button
            onClick={initialize}
            className="p-1 hover:bg-slate-100 rounded-md transition-colors"
            title="New game"
          >
            <RotateCcw size={12} />
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-2 space-y-2 fade-up">
        <section className="bg-white border border-slate-200 rounded-xl p-2.5 shadow-sm">
          <div className="flex items-center gap-2 mb-2 px-1">
            <Zap size={10} className="text-amber-500 fill-amber-500" />
            <h3 className="text-[9px] font-black uppercase text-slate-400 tracking-[0.2em]">
              Acquisition
            </h3>
          </div>

          <div className="flex flex-wrap justify-center gap-4 sm:gap-6">
            {(stateView?.factories ?? Array.from({ length: SOURCE_COUNT }, () => [])).map(
              (factory, idx) => (
                <div
                  key={idx}
                  data-role="factory"
                  data-index={idx}
                  aria-label={`Factory ${idx + 1}`}
                  onClick={() => handleSourceSelection({ kind: "factory", index: idx })}
                  className={`
                    w-16 h-16 sm:w-20 sm:h-20 rounded-full border flex flex-wrap items-center justify-center gap-1 p-2 transition-all cursor-pointer relative
                    ${
                      selection.source?.kind === "factory" &&
                      selection.source.index === idx
                        ? "border-blue-500 bg-blue-50/50 shadow-sm scale-105"
                        : "border-slate-100 bg-slate-50 hover:border-slate-300"
                    }
                    ${factory.length === 0 ? "opacity-20 grayscale pointer-events-none" : ""}
                    ${!isHumanTurn ? "pointer-events-none opacity-60" : ""}
                  `}
                >
                  {factory.length > 0 ? (
                    factory.map((t, ti) => (
                      <Node
                        key={`${t}-${ti}`}
                        type={t}
                        size="sm"
                        dataRole="tile"
                        dataColor={t}
                        ariaLabel={`${t} tile`}
                        selected={
                          selection.source?.kind === "factory" &&
                          selection.source.index === idx &&
                          selection.color === t
                        }
                        onClick={(event) => {
                          event.stopPropagation();
                          handleSourceSelection({ kind: "factory", index: idx }, t);
                        }}
                      />
                    ))
                  ) : (
                    <div className="text-[7px] font-mono text-slate-400">VOID</div>
                  )}
                </div>
              ),
            )}

            <div
              data-role="center"
              aria-label="Center pool"
              onClick={() => handleSourceSelection({ kind: "center" })}
              className={`
                min-w-[100px] sm:min-w-[140px] min-h-[60px] sm:min-h-[70px] border border-dashed rounded-2xl flex flex-wrap gap-1.5 p-2 items-center justify-center transition-all cursor-pointer
                ${
                  selection.source?.kind === "center"
                    ? "border-blue-500 bg-blue-50/50 ring-2 ring-blue-500/10"
                    : "border-slate-200 bg-slate-50/20"
                }
                ${!isHumanTurn ? "pointer-events-none opacity-60" : ""}
              `}
            >
              {stateView?.has_origin && (
                <Node
                  type="origin"
                  size="sm"
                  disabled
                  dataRole="origin"
                  ariaLabel="First player marker"
                />
              )}
              {(stateView?.center ?? []).map((t, i) => (
                <Node
                  key={`${t}-${i}`}
                  type={t}
                  size="sm"
                  dataRole="tile"
                  dataColor={t}
                  ariaLabel={`${t} tile`}
                  selected={
                    selection.source?.kind === "center" && selection.color === t
                  }
                  onClick={(event) => {
                    event.stopPropagation();
                    handleSourceSelection({ kind: "center" }, t);
                  }}
                />
              ))}
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {agents.map((agent, aIdx) => (
            <div
              key={agent.id}
              className={`bg-white border border-slate-200 rounded-xl p-3 flex flex-col relative transition-all
                ${aIdx === humanPlayer && selection.color ? "ring-2 ring-blue-500 z-10" : "shadow-sm"}
                ${aIdx === aiPlayer ? "bg-slate-50/50 opacity-95" : ""}
              `}
            >
              <div className="flex justify-between items-center mb-3">
                <div className="flex items-center gap-2">
                  <div
                    className={`p-1 rounded ${
                      aIdx === humanPlayer ? "bg-blue-600" : "bg-zinc-900"
                    } text-white`}
                  >
                    <User size={10} />
                  </div>
                  <span className="text-[10px] font-black tracking-tight uppercase leading-none">
                    {agent.name}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  {aIdx === aiPlayer && (aiThinking || aiTurn) ? (
                    <div className="flex items-center gap-1 text-[8px] font-black text-blue-500 uppercase">
                      AI Thinking
                      <span className="thinking-dot" style={{ animationDelay: "0ms" }}>
                        •
                      </span>
                      <span className="thinking-dot" style={{ animationDelay: "120ms" }}>
                        •
                      </span>
                      <span className="thinking-dot" style={{ animationDelay: "240ms" }}>
                        •
                      </span>
                    </div>
                  ) : stateView?.current_player === aIdx ? (
                    <div className="text-[8px] font-black text-blue-500 uppercase">
                      Your turn
                    </div>
                  ) : null}
                  {aIdx === aiPlayer ? (
                    <div className="flex flex-col items-end gap-1 sm:flex-row sm:items-center sm:gap-2">
                      <span
                        data-role="ai-score"
                        className="text-[10px] font-mono font-bold text-slate-600 leading-none"
                      >
                        Score {agent.score}
                      </span>
                      <div className="flex items-center gap-2 bg-slate-100 px-2 py-0.5 rounded-md border border-slate-200">
                        <SlidersHorizontal size={8} className="text-slate-400" />
                        <input
                          type="number"
                          value={mctsSims}
                          onChange={(e) =>
                            setMctsSims(Math.max(100, Number.parseInt(e.target.value, 10) || 0))
                          }
                          className="w-12 bg-transparent text-[9px] font-mono font-bold text-blue-600 outline-none"
                          step="100"
                          min={100}
                        />
                      </div>
                    </div>
                  ) : (
                    <span
                      data-role="human-score"
                      className="text-[11px] font-mono font-bold text-blue-600"
                    >
                      Score {agent.score}
                    </span>
                  )}
                </div>
              </div>

              <div className="flex flex-1 justify-between items-start gap-2">
                <div className="space-y-1.5">
                  {agent.buffers.map((buffer, bi) => {
                    const legal =
                      aIdx === humanPlayer && selection.color && legalRows.has(bi);
                    const lineView = stateView?.players[aIdx]?.pattern_lines?.[bi];
                    const capacity = lineView?.capacity ?? buffer.length;
                    const count = lineView?.count ?? buffer.filter(Boolean).length;
                    let previewCount = 0;
                    if (aIdx === humanPlayer && legal && selection.color) {
                      previewCount = Math.min(selectionCount, capacity - count);
                    }

                    return (
                      <div
                        key={bi}
                        data-role="pattern-line"
                        data-index={bi}
                        aria-label={`Pattern line ${bi + 1}`}
                        className={`flex gap-0.5 justify-end items-center p-0.5 rounded-lg transition-colors
                          ${legal ? "cursor-pointer hover:bg-blue-50" : "cursor-default"}
                          ${
                            confirmBufferIdx === bi && aIdx === humanPlayer
                              ? "bg-blue-100 outline outline-1 outline-blue-400"
                              : ""
                          }`}
                        onClick={() => aIdx === humanPlayer && handleBufferInteraction(bi)}
                      >
                        {Array.from({ length: capacity }, (_, si) => {
                          const filled = si < count && buffer[si];
                          const shouldPreview =
                            !filled &&
                            aIdx === humanPlayer &&
                            legal &&
                            si < count + previewCount;
                          return (
                            <div
                              key={si}
                              className="w-7 h-7 bg-slate-50 border border-slate-100 rounded-md flex items-center justify-center shadow-inner overflow-hidden relative"
                            >
                              {filled ? (
                                <Node
                                  type={buffer[si] as string}
                                  size="sm"
                                  disabled
                                  dataRole="tile"
                                  dataColor={buffer[si] as string}
                                  ariaLabel={`${buffer[si]} tile`}
                                />
                              ) : shouldPreview && selection.color ? (
                                <Node
                                  type={selection.color}
                                  size="sm"
                                  preview
                                  dataRole="tile-preview"
                                  dataColor={selection.color}
                                  ariaLabel={`Preview ${selection.color} tile`}
                                />
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}
                </div>

                <div className="grid grid-cols-5 gap-1 p-1.5 bg-slate-50 border border-slate-100 rounded-lg shadow-inner">
                  {agent.lattice.map((row, ri) =>
                    row.map((cell, ci) => (
                      <div
                        key={`${ri}-${ci}`}
                        className="w-7 h-7 rounded-md bg-white border border-slate-100 flex items-center justify-center relative overflow-hidden"
                      >
                        <div
                          className="absolute inset-0 opacity-[0.05]"
                          style={{
                            backgroundColor: PALETTE[LATTICE_MAP[ri][ci]],
                          }}
                        />
                        {cell && (
                          <Node
                            type={cell}
                            size="sm"
                            disabled
                            dataRole="tile"
                            dataColor={cell}
                            ariaLabel={`${cell} tile`}
                          />
                        )}
                        {!cell && <div className="w-0.5 h-0.5 rounded-full bg-slate-200" />}
                      </div>
                    )),
                  )}
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-slate-100 flex justify-between items-center">
                <div className="flex gap-0.5">
                  {LOSS_FACTORS.map((val, fi) => {
                    const isResearcher = aIdx === humanPlayer;
                    const previewTokens = selection.color
                      ? [
                          ...Array.from({ length: selectionCount }, () => selection.color),
                          ...(selectionTakesOrigin ? ["origin"] : []),
                        ]
                      : [];
                    const previewToken = previewTokens[fi - agent.inertia.length];

                    const isPreviewed =
                      isResearcher &&
                      selection.color &&
                      legalFloor &&
                      fi >= agent.inertia.length &&
                      fi < agent.inertia.length + previewTokens.length;

                    return (
                      <div key={fi} className="flex flex-col items-center gap-0.5">
                        <div
                          className={`w-6 h-6 rounded-md bg-slate-50 border border-slate-100 flex items-center justify-center shadow-inner overflow-hidden
                            ${isResearcher ? "cursor-pointer" : "cursor-default"}
                            ${
                              confirmBufferIdx === -1 && isResearcher
                                ? "bg-rose-100 outline outline-1 outline-rose-400"
                                : ""
                            }`}
                          data-role="floor-slot"
                          data-index={fi}
                          aria-label={`Floor slot ${fi + 1}`}
                          onClick={() => isResearcher && handleBufferInteraction(-1)}
                        >
                          {agent.inertia[fi] ? (
                            <Node
                              type={agent.inertia[fi]}
                              size="xs"
                              disabled
                              dataRole="tile"
                              dataColor={agent.inertia[fi]}
                              ariaLabel={`${agent.inertia[fi]} tile`}
                            />
                          ) : isPreviewed && previewToken ? (
                            <Node
                              type={previewToken}
                              size="xs"
                              preview
                              dataRole="tile-preview"
                              dataColor={previewToken}
                              ariaLabel={`Preview ${previewToken} tile`}
                            />
                          ) : null}
                        </div>
                        <span className="text-[7px] font-black text-slate-300 font-mono">
                          {val}
                        </span>
                      </div>
                    );
                  })}
                </div>
                {aIdx === humanPlayer && selection.color && legalFloor && (
                  <button
                    onClick={() => handleBufferInteraction(-1)}
                    data-role="floor-commit"
                    className={`border px-2.5 py-1.5 rounded-lg text-[8px] font-black uppercase tracking-widest transition-all ${
                      confirmBufferIdx === -1
                        ? "bg-rose-500 text-white border-rose-600 animate-pulse"
                        : "bg-rose-50 text-rose-500 border-rose-100 hover:bg-rose-100"
                    }`}
                  >
                    {confirmBufferIdx === -1 ? "Tap Again to Confirm" : "Commit to Floor"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </section>

        <section className="bg-white border border-slate-200 rounded-xl p-2.5 shadow-sm">
          <div className="flex items-center gap-2 mb-2 px-1">
            <Activity size={12} className="text-blue-600" />
            <span className="text-[9px] font-black uppercase tracking-widest text-slate-400">
              Inference
            </span>
            <span className="ml-2 inline-flex items-center gap-1 text-[8px] font-bold text-slate-400 uppercase">
              <Cpu size={10} /> {statsSnapshot.backend}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                Latency
              </span>
              <span className="text-xs font-mono font-bold">
                {formatMs(statsSnapshot.inferenceLast)}
              </span>
              <span className="text-[8px] text-slate-400">
                avg {formatMs(statsSnapshot.inferenceAvg)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                Convergence
              </span>
              <div className="h-1 bg-slate-100 rounded-full w-20 overflow-hidden mt-1">
                <div
                  className="h-full bg-blue-600 transition-all"
                  style={{ width: `${Math.round(statsSnapshot.convergence * 100)}%` }}
                />
              </div>
              <span className="text-[8px] text-slate-400">
                p95 {formatMs(statsSnapshot.inferenceP95)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                Entropy
              </span>
              <span className="text-xs font-mono font-bold text-emerald-600">
                {byNumber(statsSnapshot.policyEntropy, 3)}
              </span>
              <span className="text-[8px] text-slate-400">
                mcts {byNumber(statsSnapshot.mctsEntropy, 3)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                MCTS Nodes
              </span>
              <span className="text-xs font-mono font-bold">
                {statsSnapshot.sims.toLocaleString()}
              </span>
              <span className="text-[8px] text-slate-400">
                {statsSnapshot.simsPerSec.toLocaleString()} sims/s
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                Move Time
              </span>
              <span className="text-xs font-mono font-bold">
                {formatMs(statsSnapshot.mctsLast)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                Depth
              </span>
              <span className="text-xs font-mono font-bold">
                {statsSnapshot.depth}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                NN Evals
              </span>
              <span className="text-xs font-mono font-bold">
                {statsSnapshot.evals}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[7px] font-black text-slate-400 uppercase leading-none mb-1">
                Value Avg
              </span>
              <span className="text-xs font-mono font-bold">
                {byNumber(statsSnapshot.valueAvg, 2)}
              </span>
              <span className="text-[8px] text-slate-400">
                +/- {byNumber(statsSnapshot.valueStd, 2)}
              </span>
            </div>
          </div>

          <div className="mt-3 pt-2 border-t border-slate-100">
            <div className="text-[7px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1">
              Top Moves
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-[8px] font-mono text-slate-500">
              {statsSnapshot.topMoves.length === 0
                ? "--"
                : statsSnapshot.topMoves.map((move) => (
                    <div key={move}>{move}</div>
                  ))}
            </div>
          </div>
        </section>

        <section className="bg-zinc-900 rounded-xl p-2 text-[9px] font-mono text-slate-500 min-h-[40px]">
          {logLines.length === 0 ? (
            <p className="opacity-70 leading-tight">
              {">"} Session AX-2025.A1 active. Awaiting player signal...
            </p>
          ) : (
            logLines.map((line, idx) => (
              <p key={`${line}-${idx}`} className="opacity-80 leading-tight">
                {">"} {line}
              </p>
            ))
          )}
        </section>
      </main>

      <footer className="h-8 px-3 bg-white border-t border-slate-200 flex items-center justify-between shrink-0">
        <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
          <Target size={10} className="text-emerald-500 fill-emerald-500" />
          alpha_zero_v4_pro_stable
        </span>
        <span className="text-[8px] font-mono text-slate-400">
          {modelReady ? "Model loaded" : "Loading..."}
        </span>
      </footer>
    </div>
  );
};

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error("Root element missing");
}

createRoot(rootEl).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
