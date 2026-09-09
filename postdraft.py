#!/usr/bin/env python3
"""
postdraft.py -- grade a completed draft and emit docs/postdraft.json for the
GitHub Pages results page.

Reads the draft-tracker export (round headers, then one row per pick) and scores
every team by the projected starting lineup it can field, in value-over-
replacement terms. Also enumerates every trade of up to two players a side
between the focus team and the rest of the league, keeping the ones both sides
gain from.

    python3 postdraft.py --results hh_draft_2026.csv --focus KGB
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import statistics
from collections import defaultdict

# 12-team half-PPR starting lineup. DST is carried on the roster but excluded
# from scoring: the source projections put every defense between -69 and -190,
# which is a different scale, not a ranking we can mix into a points total.
SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1}
FLEX_POS = ("RB", "WR", "TE")
SCORED = ("QB", "RB", "WR", "TE", "K")

PLAYER_RE = re.compile(r"(?P<name>.+?)\s+(?P<pos>QB|RB|WR|TE|K|DST)\s*\|\s*(?P<nfl>[A-Z]{2,3})")


def parse_results(path, teams_per_round=12):
    """Parse the tracker export into a flat, draft-order list of picks."""
    picks, rnd = [], 0
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            head = row[0].strip()
            if head.lower().startswith("round"):
                rnd = int(head.split()[1])
                continue
            if head.lower() == "pick":
                continue
            # an asterisk prefix is the tracker's autopick marker
            m = PLAYER_RE.match(row[2].strip().lstrip("*").strip())
            if not m:
                raise ValueError(f"unparseable player cell: {row[2]!r}")
            pick = int(head)
            picks.append({
                "ovr": (rnd - 1) * teams_per_round + pick,
                "rnd": rnd,
                "pick": pick,
                "team": row[1].strip(),
                "name": m.group("name").strip(),
                "pos": m.group("pos"),
                "nfl": m.group("nfl"),
                "elapsed": row[3].strip(),
                "pos_rank": int(row[4].replace("#", "").split()[0]),
                "fpts": float(row[5]),
            })
    picks.sort(key=lambda p: p["ovr"])
    return picks


def replacement_levels(picks, n_teams=12):
    """Last league-wide starter at each position, with the FLEX slot on RB/WR."""
    by_pos = defaultdict(list)
    for p in picks:
        if p["pos"] in SCORED:
            by_pos[p["pos"]].append(p["fpts"])
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    # one FLEX per team, split 50/50 RB/WR at the margin
    depth = {"QB": SLOTS["QB"], "RB": SLOTS["RB"] + 0.5, "WR": SLOTS["WR"] + 0.5,
             "TE": SLOTS["TE"], "K": SLOTS["K"]}
    levels = {}
    for pos, per_team in depth.items():
        idx = min(int(round(per_team * n_teams)), len(by_pos[pos]) - 1)
        levels[pos] = round(by_pos[pos][idx], 1)
    return levels


def best_lineup(roster, levels):
    """Greedy-optimal starting lineup: it is optimal because slots are strictly
    ordered by position and FLEX takes the best leftover."""
    by_pos = defaultdict(list)
    for p in roster:
        if p["pos"] in SCORED:
            by_pos[p["pos"]].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: -p["fpts"])

    starters, bench = {}, []
    for pos, n in SLOTS.items():
        starters[pos] = by_pos[pos][:n]
        bench += by_pos[pos][n:]
    flex_pool = sorted((p for p in bench if p["pos"] in FLEX_POS), key=lambda p: -p["fpts"])
    starters["FLEX"] = flex_pool[:1]

    # An unfilled starting slot scores zero, which is worse than replacement —
    # never free. Without this a team can "gain" by trading away its only QB.
    empty_cost = {**{pos: levels[pos] for pos in SLOTS},
                  "FLEX": min(levels[pos] for pos in FLEX_POS)}
    n_slots = {**SLOTS, "FLEX": 1}

    vorp = {}
    for slot, ps in starters.items():
        v = sum(p["fpts"] - levels[p["pos"]] for p in ps)
        v -= empty_cost[slot] * (n_slots[slot] - len(ps))
        vorp[slot] = round(v, 1)
    pts = round(sum(p["fpts"] for ps in starters.values() for p in ps), 1)
    return starters, vorp, pts, round(sum(vorp.values()), 1)


def slim(p):
    return {k: p[k] for k in ("ovr", "rnd", "pick", "name", "pos", "nfl", "pos_rank", "fpts", "vorp")}


def trade_finder(focus, rosters, levels, min_gain=0.1, max_side=2):
    """Every package of up to `max_side` players per side between the focus team
    and the rest of the league, keeping the ones where *both* starting lineups
    improve. Bilateral gain is the whole test: a trade nobody accepts is not a
    recommendation, and in a pure-projection lens most swaps are zero-sum, so the
    survivors are exactly the positional-surplus mismatches.

    Kickers and defenses are not tradeable assets here — K spread is ~25 points
    and DST is unscored — so they are excluded from packages."""
    tradeable = lambda p: p["pos"] not in ("K", "DST")
    _, _, _, focus_base = best_lineup(rosters[focus], levels)
    out, seen = [], set()

    def packages(roster):
        pool = [p for p in roster if tradeable(p)]
        for n in range(1, max_side + 1):
            yield from itertools.combinations(pool, n)

    for other, roster in rosters.items():
        if other == focus:
            continue
        _, _, _, other_base = best_lineup(roster, levels)
        mine_pool = list(packages(rosters[focus]))
        theirs_pool = list(packages(roster))
        for give in mine_pool:
            keep_mine = [p for p in rosters[focus] if p not in give]
            for get in theirs_pool:
                keep_theirs = [p for p in roster if p not in get]
                _, _, _, mine_v = best_lineup(keep_mine + list(get), levels)
                _, _, _, theirs_v = best_lineup(keep_theirs + list(give), levels)
                d_me = round(mine_v - focus_base, 1)
                d_them = round(theirs_v - other_base, 1)
                if d_me < min_gain or d_them < min_gain:
                    continue
                key = (other, tuple(sorted(p["name"] for p in give)),
                       tuple(sorted(p["name"] for p in get)))
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "partner": other,
                    "give": [{k: p[k] for k in ("name", "pos", "pos_rank", "fpts")} for p in give],
                    "get": [{k: p[k] for k in ("name", "pos", "pos_rank", "fpts")} for p in get],
                    "gain_focus": d_me, "gain_partner": d_them,
                    "gain_total": round(d_me + d_them, 1),
                })
    out.sort(key=lambda t: (-t["gain_focus"], -t["gain_partner"]))
    return out


def prune_trades(trades, per_partner=2, total=10):
    """One page cannot show 250 permutations, and most of them are the same trade
    with a dead-weight throw-in. Collapse to the smallest package achieving a
    given (partner, gain, gain) outcome, then take the best few per partner."""
    best = {}
    size = lambda t: len(t["give"]) + len(t["get"])
    for t in trades:
        key = (t["partner"], t["gain_focus"], t["gain_partner"])
        if key not in best or size(t) < size(best[key]):
            best[key] = t

    kept, count = [], defaultdict(int)
    for t in sorted(best.values(), key=lambda t: (-t["gain_focus"], size(t), -t["gain_partner"])):
        if count[t["partner"]] >= per_partner:
            continue
        count[t["partner"]] += 1
        kept.append(t)
        if len(kept) >= total:
            break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="hh_draft_2026.csv")
    ap.add_argument("--focus", default="KGB")
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--label", default="HH 2026 half-PPR, 12-team snake")
    args = ap.parse_args()

    picks = parse_results(args.results, args.teams)
    levels = replacement_levels(picks, args.teams)
    for p in picks:
        p["vorp"] = round(p["fpts"] - levels[p["pos"]], 1) if p["pos"] in SCORED else 0.0

    rosters = defaultdict(list)
    for p in picks:
        rosters[p["team"]].append(p)

    teams = []
    for name, roster in rosters.items():
        starters, vorp, pts, total = best_lineup(roster, levels)
        starter_ids = {id(p) for ps in starters.values() for p in ps}
        teams.append({
            "team": name,
            "pts": pts,
            "vorp": total,
            "pos_vorp": vorp,
            "starters": {slot: [slim(p) for p in ps] for slot, ps in starters.items()},
            "bench": [slim(p) for p in sorted(roster, key=lambda p: -p["vorp"])
                      if id(p) not in starter_ids],
            "bench_vorp": round(sum(p["vorp"] for p in roster
                                    if id(p) not in starter_ids and p["pos"] in SCORED), 1),
            "roster": [slim(p) for p in sorted(roster, key=lambda p: p["ovr"])],
            "dead_spots": sum(1 for p in roster if p["pos"] in SCORED and p["vorp"] < 0
                              and id(p) not in starter_ids),
        })
    teams.sort(key=lambda t: -t["vorp"])
    for i, t in enumerate(teams, 1):
        t["rank"] = i

    slots = list(SLOTS) + ["FLEX"]
    median = {s: round(statistics.median(t["pos_vorp"][s] for t in teams), 1) for s in slots}
    pos_rank = {}
    for s in slots:
        order = sorted(teams, key=lambda t: -t["pos_vorp"][s])
        for i, t in enumerate(order, 1):
            pos_rank.setdefault(t["team"], {})[s] = i
    for t in teams:
        t["pos_rank"] = pos_rank[t["team"]]

    # what the focus team passed on: best VORP still on the board at each of its picks
    focus_log = []
    for p in picks:
        if p["team"] != args.focus:
            continue
        board = sorted((q for q in picks if q["ovr"] >= p["ovr"] and q["pos"] in SCORED),
                       key=lambda q: -q["vorp"])
        focus_log.append({**slim(p),
                          "best_avail": [{"name": q["name"], "pos": q["pos"], "vorp": q["vorp"],
                                          "ovr": q["ovr"], "team": q["team"]}
                                         for q in board[:3]]})

    all_trades = trade_finder(args.focus, rosters, levels)

    out = {
        "meta": {
            "label": args.label,
            "teams": args.teams,
            "rounds": max(p["rnd"] for p in picks),
            "focus": args.focus,
            "lineup": {**SLOTS, "FLEX": 1},
            "replacement": levels,
            "median_pos_vorp": median,
            "dst_excluded": True,
        },
        "teams": teams,
        "picks": [dict(p) for p in picks],
        "focus_log": focus_log,
        "trades": prune_trades(all_trades),
        "trades_found": len(all_trades),
    }

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "postdraft.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    if os.path.abspath(args.results) != os.path.abspath(os.path.join(args.outdir, os.path.basename(args.results))):
        shutil.copyfile(args.results, os.path.join(args.outdir, os.path.basename(args.results)))

    print(f"wrote {args.outdir}/postdraft.json — replacement {levels}")
    for t in teams:
        flag = "  <-- focus" if t["team"] == args.focus else ""
        print(f"  {t['rank']:>2}. {t['team']:<11} {t['pts']:>7.1f} pts  VORP {t['vorp']:>7.1f}{flag}")
    print(f"  {len(all_trades)} mutually-beneficial trades found for {args.focus}; kept {len(out['trades'])}")


if __name__ == "__main__":
    raise SystemExit(main())
