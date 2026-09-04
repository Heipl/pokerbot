"""Attribute a bot's net result to the street a hand ended on and its last
action there, so losing spots show up instead of having to be guessed.

    python tools/leak_finder.py --file gamelog.txt --hero player1 --worst 10
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

STREET_MARKERS = [
    ("preflop", re.compile(r"^Flop \[")),
    ("flop", re.compile(r"^Discard 1 \[")),
    ("discard1", re.compile(r"^Discard 2 \[")),
    ("discard2", re.compile(r"^Turn \[")),
    ("turn", re.compile(r"^River \[")),
]
ORDER = ["preflop", "flop", "discard1", "discard2", "turn", "river"]


def parse_rounds(text: str):
    for chunk in re.split(r"\nRound #", text)[1:]:
        yield chunk.splitlines()


def analyse(lines, hero):
    """Return (street_reached, hero_last_action, delta, opp_last_action)."""
    street = "preflop"
    hero_actions = []
    opp_actions = []
    delta = None
    for ln in lines:
        ln = ln.strip()
        advanced = False
        for name, rx in STREET_MARKERS:
            if rx.match(ln):
                idx = ORDER.index(name)
                street = ORDER[min(idx + 1, len(ORDER) - 1)]
                advanced = True
                break
        if advanced:
            continue
        m = re.match(r"(\w+) (folds|calls|checks|bets|raises|discards)\b", ln)
        if m:
            who, act = m.group(1), m.group(2)
            if act == "discards":
                continue
            (hero_actions if who == hero else opp_actions).append((street, act))
            continue
        m = re.match(r"(\w+) awarded (-?\d+)", ln)
        if m and m.group(1) == hero:
            delta = int(m.group(2))
    if delta is None:
        return None
    last_hero = hero_actions[-1][1] if hero_actions else "none"
    last_opp = opp_actions[-1][1] if opp_actions else "none"
    return street, last_hero, delta, last_opp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--hero", default="player1")
    ap.add_argument("--worst", type=int, default=0,
                    help="also list the N single worst rounds")
    args = ap.parse_args()

    text = open(args.file, encoding="utf-8", errors="replace").read()
    by_street = defaultdict(lambda: {"n": 0, "chips": 0, "won": 0})
    by_action = defaultdict(lambda: {"n": 0, "chips": 0})
    by_pair = defaultdict(lambda: {"n": 0, "chips": 0})
    worst = []
    total = 0

    for lines in parse_rounds(text):
        got = analyse(lines, args.hero)
        if not got:
            continue
        street, act, delta, opp_act = got
        total += delta
        s = by_street[street]
        s["n"] += 1
        s["chips"] += delta
        s["won"] += int(delta > 0)
        a = by_action[act]
        a["n"] += 1
        a["chips"] += delta
        p = by_pair[(street, act)]
        p["n"] += 1
        p["chips"] += delta
        worst.append((delta, street, act, opp_act))

    if not by_street:
        print("no rounds parsed - check --hero matches the log")
        return 1

    print(f"hero = {args.hero}    net = {total:+d} chips over "
          f"{sum(v['n'] for v in by_street.values())} rounds\n")

    print(f"{'street hand ENDED on':<22}{'n':>6}{'net chips':>12}{'chips/hand':>12}{'win%':>7}")
    for st in ORDER:
        v = by_street.get(st)
        if not v or not v["n"]:
            continue
        print(f"{st:<22}{v['n']:>6}{v['chips']:>12}{v['chips']/v['n']:>12.2f}"
              f"{v['won']/v['n']:>7.0%}")

    print(f"\n{'hero last action':<22}{'n':>6}{'net chips':>12}{'chips/hand':>12}")
    for act, v in sorted(by_action.items(), key=lambda kv: kv[1]["chips"]):
        print(f"{act:<22}{v['n']:>6}{v['chips']:>12}{v['chips']/v['n']:>12.2f}")

    print("\nbiggest leaks (street + hero action, ranked by total chips lost):")
    ranked = sorted(by_pair.items(), key=lambda kv: kv[1]["chips"])[:8]
    for (st, act), v in ranked:
        print(f"   {st:<10} {act:<8} n={v['n']:>4}  net={v['chips']:>8}  "
              f"per hand={v['chips']/v['n']:>8.2f}")

    if args.worst:
        print(f"\n{args.worst} single worst rounds:")
        for delta, st, act, opp in sorted(worst)[:args.worst]:
            print(f"   {delta:>7}  ended on {st:<9} hero {act:<7} opp {opp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
