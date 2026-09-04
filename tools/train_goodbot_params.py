"""Tune GoodBot's numeric thresholds by hill-climbing on match results.

Write a candidate parameter JSON, run the engine for a match, parse the final
bankroll from gamelog.txt, keep improvements. Dependency-free on purpose.

  python tools/train_goodbot_params.py --iters 30 --sigma 0.15

Forces the pool to GoodBot via POOL_OPPONENT. The score is the bankroll of
whichever seat opponent_pool occupies, resolved from config.py -- hardcoding it
scored GoodBot's opponent once config.py was repointed, so every accepted
mutation made GoodBot worse.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ENGINE_DIR = Path(__file__).resolve().parent.parent
OPPONENT_POOL_DIR = ENGINE_DIR / "opponent_pool"
DEFAULT_PARAMS_PATH = OPPONENT_POOL_DIR / "goodbot_params.json"
GAMELOG_PATH = ENGINE_DIR / "gamelog.txt"

# Which seat opponent_pool occupies -- resolved, never assumed, so repointing
# PLAYER_1_PATH cannot silently invert the objective.
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
try:
    from config import (PLAYER_1_NAME, PLAYER_2_NAME,
                        PLAYER_1_PATH, PLAYER_2_PATH)
    if "opponent_pool" in str(PLAYER_1_PATH):
        SCORED_NAME = PLAYER_1_NAME
    elif "opponent_pool" in str(PLAYER_2_PATH):
        SCORED_NAME = PLAYER_2_NAME
    else:
        raise RuntimeError(
            "train_goodbot_params tunes GoodBot, which lives in opponent_pool, "
            "but neither PLAYER_1_PATH nor PLAYER_2_PATH points there: "
            f"{PLAYER_1_PATH!r} / {PLAYER_2_PATH!r}")
except ImportError as _e:
    # No "player1" fallback: that default is exactly the inverted objective
    # this resolution exists to remove.
    raise RuntimeError(
        "train_goodbot_params needs PLAYER_1/2_NAME and PLAYER_1/2_PATH from "
        "config.py to know which seat GoodBot occupies: {}".format(_e))


@dataclass
class Params:
    # Must match the keys GoodBot.Params reads in opponent_pool/player.py.

    call_equity_multiplier: float = 1.0

    facing_bet_raise_pot_lo: float = 0.5
    facing_bet_raise_pot_hi: float = 1.0

    p_fold_base: float = 0.0
    p_fold_equity_mult: float = 0.4

    raise_if_ev_margin: float = 0.0
    raise_bluff_max_equity: float = 0.4
    raise_bluff_prob: float = 0.2

    bet_value_threshold: float = 0.65
    bet_bluff_max_equity: float = 0.45
    bet_bluff_prob: float = 0.15

    unchecked_bet_pot_lo: float = 0.5
    unchecked_bet_pot_hi: float = 0.75

    # Equity thresholds picking the raise size when facing a bet. Defaults are
    # GoodBot's current hardcoded values, so a run starts from live behaviour.
    raise_size_equity_hi: float = 0.62
    raise_size_equity_mid: float = 0.54

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        if not isinstance(d, dict):
            return cls()
        fields = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        kwargs = {k: d[k] for k in fields if k in d}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_params(path: Path) -> Params:
    try:
        return Params.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return Params()


def save_params(path: Path, params: Params) -> None:
    path.write_text(json.dumps(params.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def mutate(best: Params, rng: random.Random, sigma: float) -> Params:
    d = best.to_dict()

    # Only mutate floats (and a couple ints) to keep search stable.
    float_keys = [
        "call_equity_multiplier",
        "facing_bet_raise_pot_lo",
        "facing_bet_raise_pot_hi",
        "p_fold_base",
        "p_fold_equity_mult",
        "raise_if_ev_margin",
        "raise_bluff_max_equity",
        "raise_bluff_prob",
        "bet_value_threshold",
        "bet_bluff_max_equity",
        "bet_bluff_prob",
        "unchecked_bet_pot_lo",
        "unchecked_bet_pot_hi",
        "raise_size_equity_hi",
        "raise_size_equity_mid",
    ]

    for k in float_keys:
        v = float(d[k])
        v = v + rng.gauss(0.0, sigma)
        d[k] = v

    d["call_equity_multiplier"] = clamp(float(d["call_equity_multiplier"]), 0.5, 2.0)

    lo = clamp(float(d["facing_bet_raise_pot_lo"]), 0.0, 5.0)
    hi = clamp(float(d["facing_bet_raise_pot_hi"]), 0.0, 5.0)
    if lo > hi:
        lo, hi = hi, lo
    d["facing_bet_raise_pot_lo"], d["facing_bet_raise_pot_hi"] = lo, hi

    lo = clamp(float(d["unchecked_bet_pot_lo"]), 0.0, 5.0)
    hi = clamp(float(d["unchecked_bet_pot_hi"]), 0.0, 5.0)
    if lo > hi:
        lo, hi = hi, lo
    d["unchecked_bet_pot_lo"], d["unchecked_bet_pot_hi"] = lo, hi

    d["p_fold_base"] = clamp(float(d["p_fold_base"]), 0.0, 1.0)
    d["p_fold_equity_mult"] = clamp(float(d["p_fold_equity_mult"]), 0.0, 2.0)

    d["raise_bluff_max_equity"] = clamp(float(d["raise_bluff_max_equity"]), 0.0, 1.0)
    d["raise_bluff_prob"] = clamp(float(d["raise_bluff_prob"]), 0.0, 1.0)
    d["bet_value_threshold"] = clamp(float(d["bet_value_threshold"]), 0.0, 1.0)
    d["bet_bluff_max_equity"] = clamp(float(d["bet_bluff_max_equity"]), 0.0, 1.0)
    d["bet_bluff_prob"] = clamp(float(d["bet_bluff_prob"]), 0.0, 1.0)

    # 'hi' must stay above 'mid' or the size ladder inverts.
    hi = clamp(float(d["raise_size_equity_hi"]), 0.0, 1.0)
    mid = clamp(float(d["raise_size_equity_mid"]), 0.0, 1.0)
    if mid > hi:
        hi, mid = mid, hi
    d["raise_size_equity_hi"], d["raise_size_equity_mid"] = hi, mid

    return Params.from_dict(d)


_FINAL_RE = re.compile(r"^Final(?P<rest>.*)$")
_BANKROLL_RE = re.compile(r"\(([-+]?\d+)\)")


def parse_score_from_gamelog(path: Path) -> int:
    """Returns the bankroll of the seat opponent_pool occupies (SCORED_NAME)."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    final_line = None
    for line in reversed(txt.splitlines()):
        if line.startswith("Final"):
            final_line = line
            break
    if final_line is None:
        raise RuntimeError("Could not find Final line in gamelog")

    # The engine swaps seats every round, so the order of the Final line depends
    # on the parity of NUM_ROUNDS. Match by name, not position.
    m = re.search(r"\b" + re.escape(SCORED_NAME) + r"\b[^(]*\(([-+]?\d+)\)", final_line)
    if m:
        return int(m.group(1))
    nums = _BANKROLL_RE.findall(final_line)
    if len(nums) < 1:
        raise RuntimeError(f"Could not parse bankrolls from: {final_line}")
    raise RuntimeError(
        f"Found bankrolls but no entry named {SCORED_NAME!r} in: {final_line}")


def run_engine_once(extra_env: dict[str, str], timeout_s: float) -> int:
    env = dict(os.environ)
    env.update(extra_env)

    # Make runs reproducible-ish
    env.setdefault("POOL_OPPONENT", "GoodBot")
    env.setdefault("GOODBOT_PARAMS_PATH", str(DEFAULT_PARAMS_PATH))
    # Freeze the opponent: neural_cfr_bot trains by default, and a moving
    # opponent turns every accept/reject into noise. setdefault, so it is opt-in.
    env.setdefault("NCFR_TRAINING", "0")

    # Drop the old gamelog so a crashed run cannot be scored on stale results.
    try:
        GAMELOG_PATH.unlink(missing_ok=True)  # py3.8+: missing_ok
    except TypeError:
        if GAMELOG_PATH.exists():
            GAMELOG_PATH.unlink()

    proc = subprocess.run(
        [sys.executable, "engine.py"],
        cwd=str(ENGINE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"engine.py exited with {proc.returncode}\n{proc.stdout[-4000:]}")

    if not GAMELOG_PATH.exists():
        raise RuntimeError("engine.py did not produce gamelog.txt")

    return parse_score_from_gamelog(GAMELOG_PATH)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=999999.0)
    ap.add_argument("--params", type=str, default=str(DEFAULT_PARAMS_PATH))
    args = ap.parse_args()

    params_path = Path(args.params)

    rng = random.Random(args.seed)

    best = load_params(params_path)
    save_params(params_path, best)

    print(f"[train] engine_dir={ENGINE_DIR}")
    print(f"[train] params_path={params_path}")
    print(f"[train] iters={args.iters} sigma={args.sigma} seed={args.seed}")

    best_score = run_engine_once({}, timeout_s=args.timeout)
    print(f"[train] baseline score={best_score}")

    for i in range(1, args.iters + 1):
        cand = mutate(best, rng, sigma=args.sigma)
        save_params(params_path, cand)

        t0 = time.time()
        score = run_engine_once({}, timeout_s=args.timeout)
        dt = time.time() - t0

        improved = score > best_score
        tag = "ACCEPT" if improved else "reject"
        print(f"[train] iter={i:03d} score={score:>6} best={best_score:>6} ({tag})  time={dt:.1f}s")

        if improved:
            best, best_score = cand, score
            save_params(params_path, best)

    print(f"[train] done best_score={best_score}")
    print(f"[train] best_params saved to {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
