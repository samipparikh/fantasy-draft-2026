#!/usr/bin/env python3
"""
build_site.py -- run the draft model and emit docs/data.json for the GitHub Pages
site. Imports draft_model as a library so the site and the CLI report can never
disagree about the numbers.

    python3 build_site.py --players ringer_2026.csv --sims 4000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics

import draft_model as dm


def expected_best(surv_at_pick, players, positions=None):
    """E[max VOR] over the board at one pick, given availability probabilities."""
    cands = sorted(
        ((pa, players[idx]) for idx, pa in surv_at_pick.items() if pa > 0.0),
        key=lambda t: -t[1].vor,
    )
    e, carry = 0.0, 1.0
    for pa, p in cands:
        if positions and p.pos not in positions:
            continue
        e += carry * pa * p.vor
        carry *= 1.0 - pa
        if carry < 1e-4:
            break
    return e


def player_row(p, pa=None, take=None):
    d = {"name": p.name, "pos": p.pos, "tier": p.tier, "team": p.team,
         "adp": round(p.adp, 1), "vor": round(p.vor, 1), "av": round(p.av)}
    if pa is not None:
        d["p_avail"] = round(pa, 3)
    if take is not None:
        d["p_take"] = round(take, 3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="ringer_2026.csv")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--slots", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--plan-rounds", type=int, default=8)
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    starters, flex = dm.parse_starters("QB:1,RB:2,WR:2,TE:1,FLEX:1,K:1,DST:1")
    lg = dm.League(args.teams, args.rounds, starters, flex, 0.20, 15.0)

    cfg = dm.Cfg()
    cfg.sims = args.sims
    cfg.rb_weight = 1.15
    cfg.vona_weight = 0.6
    cfg.candidates_per_pos = 4
    cfg.vona_depth = 25
    cfg.track_avail = True
    cfg.track_top = 24
    cfg.plan_rounds = args.plan_rounds

    players, skipped, derived, value_src = dm.load_players(args.players)
    base = dm.compute_vor(players, lg)
    dm.assign_tiers(players, lg)

    out = {
        "meta": {
            "teams": lg.teams, "rounds": lg.rounds, "sims": cfg.sims,
            "scoring": "half PPR (0.5/rec)",
            "starters": {**{k: v for k, v in lg.starters.items()}, "FLEX": lg.flex},
            "pool_size": len(players),
            "value_source": value_src,
            "rb_weight": cfg.rb_weight,
            "adp_noise": lg.sigma_frac,
            "source": "The Ringer 2026 preseason rankings (Kelly / Heifetz / Horlbeck), "
                      "half-PPR draft tracker, updated Aug 3 2026",
            "priced_players": sum(1 for p in players if p.av),
        },
        "baselines": {pos: {"rank": f"{pos}{r}", "pts": round(v, 1)}
                      for pos, (v, r) in base.items()},
        "tiers": {},
        "slots": {},
        "strategy_doc": dm.STRATEGY_DOC,
    }

    # ---- tier structure per position (for the cliff small-multiples) ----
    for pos in ("RB", "WR", "TE", "QB"):
        pool = sorted([p for p in players if p.pos == pos], key=lambda p: -p.vor)[:14]
        out["tiers"][pos] = [player_row(p) for p in pool]

    # ---- per-slot simulation ----
    for slot in args.slots:
        picks = lg.pick_numbers(slot)
        base_res, _ = dm.run_sims(players, lg, cfg, slot, "rb_priority",
                                  surv=None, seed=args.seed + slot)
        surv = dm.survival_table(base_res, picks)

        results = {}
        for k, strat in enumerate(dm.STRATEGIES):
            res, _ = dm.run_sims(players, lg, cfg, slot, strat, surv=surv,
                                 seed=args.seed + 100 * (k + 1) + slot)
            results[strat] = res

        strat_out = {}
        for name, res in results.items():
            pts = sorted(res.lineup_pts)
            strat_out[name] = {
                "mean": round(statistics.fmean(pts), 1),
                "p10": round(pts[int(0.10 * (len(pts) - 1))], 1),
                "p50": round(pts[len(pts) // 2], 1),
                "p90": round(pts[int(0.90 * (len(pts) - 1))], 1),
                "rb": round(res.roster_pos["RB"] / max(1, res.n), 1),
                "wr": round(res.roster_pos["WR"] / max(1, res.n), 1),
            }

        plan_res = results["rb_priority"]
        plan = []
        for i, pk in enumerate(picks):
            rnd = lg.round_of(pk)
            if rnd > cfg.plan_rounds:
                break
            mix = plan_res.pos_taken[i]
            tot = max(1, sum(mix.values()))
            targets = [player_row(players[idx], surv[i].get(idx, 0.0), n / tot)
                       for idx, n in plan_res.player_taken[i].most_common(6)]
            plan.append({
                "round": rnd,
                "pick": f"{rnd}.{(pk - 1) % lg.teams + 1:02d}",
                "overall": pk,
                "mix": [{"pos": pos, "share": round(n / tot, 3)}
                        for pos, n in mix.most_common(5)],
                "targets": targets,
            })

        out["slots"][str(slot)] = {
            "slot": slot,
            "picks": [{"round": lg.round_of(pk), "overall": pk,
                       "label": f"{lg.round_of(pk)}.{(pk - 1) % lg.teams + 1:02d}"}
                      for pk in picks],
            "turn_gaps": {"r1_r2": picks[1] - picks[0], "r2_r3": picks[2] - picks[1]},
            "strategies": strat_out,
            "ev": {
                "r1": round(expected_best(surv[0], players), 1),
                "r2": round(expected_best(surv[1], players), 1),
            },
            "plan": plan,
        }

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "data.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    # ship the raw artifacts alongside the page
    for src, dst in (("ringer_2026.csv", "ringer_2026.csv"),
                     ("out/board_vor.csv", "board_vor.csv"),
                     ("out/report.txt", "report.txt")):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.outdir, dst))
    for slot in args.slots:
        s = f"out/availability_slot{slot}.csv"
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(args.outdir, f"availability_slot{slot}.csv"))

    print(f"wrote {args.outdir}/data.json")
    for slot in args.slots:
        d = out["slots"][str(slot)]
        print(f"  slot {slot}: rb_priority mean {d['strategies']['rb_priority']['mean']}, "
              f"EV R1 {d['ev']['r1']} R2 {d['ev']['r2']}")


if __name__ == "__main__":
    raise SystemExit(main())
