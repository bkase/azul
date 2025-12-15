Your run is **no longer “stuck/infinite”** (moves/g ≈ 65, trunc=0%), so the *worst* failure mode is gone. But the new metrics show you’re still learning something very wrong:

* **floor% is rising**: 36% → 44% → 49%
* and **most of those floor moves are optional** (opt_floor ≈ 92%), i.e. *there existed a non-floor legal action, but it still chose Floor*.

Roughly, at iter 20 you’re at ~0.493 × 0.919 ≈ **45% of all moves are “optional floor.”** That’s wildly non-Azul.

This points to the same underlying issue: **floor is “cheap” in the short horizon because its penalty is delayed**, so MCTS (and then the net) doesn’t get a clean negative signal early enough.

### Why this happens in your code

* Choosing Floor doesn’t change score immediately: `add_to_floor` just appends tokens. 
* Floor penalties are applied later in `resolve_end_of_round`. 
* But your per-move reward for training is computed from **per-step score deltas** (DenseScoreDelta), and then turned into a zero-sum delta.
  So early in a round, “dump to floor” often looks like ~0 immediate reward, even though it’s objectively terrible.

---

## What to do next (in order)

### 1) Stop this run and fix the reward timing for floor

This is the highest-leverage patch:

**Make floor penalty apply immediately when a tile is placed into the floor line**, but keep the *final score the same* by removing the score subtraction at end-of-round.

Concretely:

* In `apply_action`, inside `add_to_floor`, when you place a token into floor slot `k`, immediately do `player.score += FLOOR_PENALTY[k]`. (You already know `k` because it’s `player.floor.len` before increment.)
* In `resolve_end_of_round`, keep discarding the floor tiles and detecting the first-player marker, but **do not modify score there**. 

This causes the **DenseScoreDelta reward to fire immediately** when the agent dumps to floor, which makes MCTS avoid it long before the net has learned sophisticated long-horizon value.

> This change alone typically collapses optional-floor% dramatically.

### 2) Add two tests that should fail today and pass after the patch

You already have floor-penalty tests that assume penalties happen in `resolve_end_of_round`. You’ll need to update them.

Add:

**(A) Immediate penalty test**

* Set up a state where a move puts exactly 1 tile into an empty floor.
* Call `apply_action` once.
* Assert the acting player’s score decreased by `FLOOR_PENALTY[0]` immediately.

This will fail today because score doesn’t change until end-of-round.

**(B) No double-penalty test**

* Create a state where floor has some tiles (or you add them via a move), then force end-of-round.
* Assert that `resolve_end_of_round` does **not** change the score further (only discards / marker handling).

This catches “oops, we penalized twice.” 

Also update or rewrite `test_floor_penalty_allows_negative_scores` because it currently expects `resolve_end_of_round` to apply -14. 

### 3) Improve your debugging signal: log **floor mass in the MCTS policy**, not just chosen moves

Right now you log “chose floor.” But if MCTS’s returned policy is already 60% floor-mass, that’s different from “policy is fine but sampling picks floor.”

In `self_play_game`, you already have `search_result.policy` (a full ACTION_SPACE_SIZE distribution).
Compute:

* `policy_floor_mass = sum(pi[a] for legal a where decode(a).dest==Floor)`
* log its mean per game/iter

This tells you whether your fix is changing *search preference* vs *sampling noise*.

### 4) Raise `temp_cutoff_move`

Default cutoff is 30. 
Your games are ~65 moves, so half the game is already argmax. That can “lock in” garbage early.

For now, set it to something like **200** (i.e. “always tau=1 for entire game” in practice). This makes early training far less brittle.

---

## After you implement (1) + (4), rerun the same training config

Keep your `mcts-sims 300` for now.

What you’re looking for in the log:

* floor% should fall hard (you should not see it climbing toward 50%)
* opt_floor% should fall hard (ideally into the single digits or teens)
* wall should stop trending downward (should stabilize or increase)

If it *doesn’t* improve, then we go to the next tier: “restrict optional floor moves from the action space temporarily” (a learning scaffold) or bootstrap truncated tails — but I’d bet you won’t need that once floor penalties are immediate.

If you paste one line of logs after the patch (iter 0 / 10 / 20 again), I can tell you immediately whether we’ve killed the attractor.
