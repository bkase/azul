/* tslint:disable */
/* eslint-disable */

export class GameStateHandle {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  clone_handle(): GameStateHandle;
  current_player(): number;
  is_game_over(): boolean;
  round(): number;
  scores(): Int16Array;
  legal_action_ids(): Uint16Array;
  legal_action_strings(): string[];
  action_id_to_string(action_id: number): string;
  encode_observation(player: number): Float32Array;
  render_text(highlight_player?: number | null): string;
  state_view(): any;
  legal_action_details(): any;
  apply_action_id(action_id: number): any;
}

export function action_space_size(): number;

export function new_game_state(seed: bigint): GameStateHandle;

export function observation_size(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_gamestatehandle_free: (a: number, b: number) => void;
  readonly action_space_size: () => number;
  readonly observation_size: () => number;
  readonly new_game_state: (a: bigint) => number;
  readonly gamestatehandle_clone_handle: (a: number) => number;
  readonly gamestatehandle_current_player: (a: number) => number;
  readonly gamestatehandle_is_game_over: (a: number) => number;
  readonly gamestatehandle_round: (a: number) => number;
  readonly gamestatehandle_scores: (a: number) => [number, number];
  readonly gamestatehandle_legal_action_ids: (a: number) => [number, number];
  readonly gamestatehandle_legal_action_strings: (a: number) => [number, number];
  readonly gamestatehandle_action_id_to_string: (a: number, b: number) => [number, number];
  readonly gamestatehandle_encode_observation: (a: number, b: number) => [number, number];
  readonly gamestatehandle_render_text: (a: number, b: number) => [number, number];
  readonly gamestatehandle_state_view: (a: number) => [number, number, number];
  readonly gamestatehandle_legal_action_details: (a: number) => [number, number, number];
  readonly gamestatehandle_apply_action_id: (a: number, b: number) => [number, number, number];
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_drop_slice: (a: number, b: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
