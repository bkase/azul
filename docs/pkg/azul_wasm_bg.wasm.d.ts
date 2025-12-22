/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_gamestatehandle_free: (a: number, b: number) => void;
export const action_space_size: () => number;
export const observation_size: () => number;
export const new_game_state: (a: bigint) => number;
export const gamestatehandle_clone_handle: (a: number) => number;
export const gamestatehandle_current_player: (a: number) => number;
export const gamestatehandle_is_game_over: (a: number) => number;
export const gamestatehandle_round: (a: number) => number;
export const gamestatehandle_scores: (a: number) => [number, number];
export const gamestatehandle_legal_action_ids: (a: number) => [number, number];
export const gamestatehandle_legal_action_strings: (a: number) => [number, number];
export const gamestatehandle_action_id_to_string: (a: number, b: number) => [number, number];
export const gamestatehandle_encode_observation: (a: number, b: number) => [number, number];
export const gamestatehandle_render_text: (a: number, b: number) => [number, number];
export const gamestatehandle_state_view: (a: number) => [number, number, number];
export const gamestatehandle_legal_action_details: (a: number) => [number, number, number];
export const gamestatehandle_apply_action_id: (a: number, b: number) => [number, number, number];
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __externref_drop_slice: (a: number, b: number) => void;
export const __externref_table_dealloc: (a: number) => void;
export const __wbindgen_start: () => void;
