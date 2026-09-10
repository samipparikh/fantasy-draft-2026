#!/usr/bin/env python3
"""
dst_model.py -- project every defense's fantasy points for all 18 weeks of 2026
by opponent, and pick a start each week for one roster. Emits docs/dst.json.

    python3 dst_model.py --refresh              # re-pull schedule + 2025 form
    python3 dst_model.py --rostered TB,NYG,LV

The model is two components, both interpretable:

  1. POINTS ALLOWED. An expected points-allowed line for each matchup, from an
     offense index for the opponent, a defense index for the defense, and home
     field. Its three coefficients are fit by least squares against the closing
     market lines for the games that have them, so the scale is the market's, not
     a guess. Expected tier points are then integrated over a normal centred on
     that line — a defense projected to allow 20 is not a defense that allows
     exactly 20, and the tier table is steeply non-linear, so the integral and
     the point estimate are different numbers.
  2. BIG PLAYS. Sacks, takeaways and return scores, as a linear function of the
     same two indices around a league-average baseline.

Nothing here is play-by-play. It is a matchup model: it ranks weeks for a
defense, and defenses within a week, which is all a streaming decision needs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import urllib.request
from collections import defaultdict

NFLVERSE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DATA_DIR = "data"
SCHEDULE_CSV = os.path.join(DATA_DIR, "nfl_2026_schedule.csv")
FORM_CSV = os.path.join(DATA_DIR, "nfl_2025_team_form.csv")

# the draft export's abbreviations vs nflverse's
ALIAS = {"LAR": "LA", "JAC": "JAX"}

# Standard points-allowed tiers (ESPN/Yahoo default). Edit here if the league
# scores defenses differently — everything downstream follows.
PA_TIERS = [(0, 0, 10), (1, 6, 7), (7, 13, 4), (14, 20, 1),
            (21, 27, 0), (28, 34, -1), (35, 999, -4)]

# league-average per-game fantasy points from sacks + takeaways + return scores:
# ~2.4 sacks, ~0.8 INT, ~0.6 fumble recoveries, ~0.15 defensive TDs
BIGPLAY_BASE = 6.1
BIGPLAY_D = 1.5   # points per SD of defense quality
BIGPLAY_O = -0.7  # points per SD of opponent offence

TEAM_NAME = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}


# ---------------------------------------------------------------- data refresh

def refresh(season=2026, prior=2025):
    """Pull nflverse games.csv and vendor the two slices this model needs, so a
    normal run is offline and reproducible."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with urllib.request.urlopen(NFLVERSE, timeout=60) as fh:
        rows = list(csv.DictReader(line.decode("utf-8") for line in fh))

    sched = [g for g in rows if g["season"] == str(season) and g["game_type"] == "REG"]
    with open(SCHEDULE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["week", "gameday", "away_team", "home_team", "spread_line", "total_line"])
        for g in sorted(sched, key=lambda g: (int(g["week"]), g["gameday"])):
            w.writerow([g["week"], g["gameday"], g["away_team"], g["home_team"],
                        g["spread_line"], g["total_line"]])

    # prior-season scoring form, plus the residual spread of actual points around
    # the market's implied points — that is the sigma the tier integral needs
    done = [g for g in rows if g["season"] == str(prior) and g["game_type"] == "REG" and g["home_score"]]
    agg = defaultdict(lambda: [0, 0, 0])
    resid = []
    for g in done:
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        for t, pf, pa in ((g["home_team"], hs, as_), (g["away_team"], as_, hs)):
            a = agg[t]
            a[0] += pf
            a[1] += pa
            a[2] += 1
        if g["total_line"] and g["spread_line"]:
            tot, spr = float(g["total_line"]), float(g["spread_line"])
            resid += [hs - (tot + spr) / 2, as_ - (tot - spr) / 2]
    with open(FORM_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team", "games", "pts_for_pg", "pts_against_pg"])
        for t in sorted(agg):
            pf, pa, n = agg[t]
            w.writerow([t, n, round(pf / n, 3), round(pa / n, 3)])
        w.writerow(["_RESIDUAL_SD", len(resid), round(statistics.pstdev(resid), 3), ""])
    print(f"wrote {SCHEDULE_CSV} ({len(sched)} games) and {FORM_CSV} "
          f"({len(agg)} teams, residual sd {statistics.pstdev(resid):.2f})")


# ---------------------------------------------------------------- inputs

def load_schedule():
    games, weeks = [], set()
    with open(SCHEDULE_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = {"week": int(r["week"]), "gameday": r["gameday"],
                 "away": r["away_team"], "home": r["home_team"],
                 "spread": float(r["spread_line"]) if r["spread_line"] else None,
                 "total": float(r["total_line"]) if r["total_line"] else None}
            games.append(g)
            weeks.add(g["week"])
    return games, sorted(weeks)


def load_form():
    form, resid_sd = {}, 10.0
    with open(FORM_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["team"] == "_RESIDUAL_SD":
                resid_sd = float(r["pts_for_pg"])
                continue
            form[r["team"]] = {"pf": float(r["pts_for_pg"]), "pa": float(r["pts_against_pg"])}
    return form, resid_sd


def draft_offence(results_csv):
    """2026 offence proxy per NFL team: the projected points of the skill players
    this league actually drafted from it — best QB, top 2 RB, top 3 WR, best TE.
    Fixed slot counts so a team nobody drafted deep is not punished for it.

    Also returns the tracker's DST projections and their DEF ranks. The
    projections are negative on some season-points-allowed scale, but they are
    monotone in the rank (DEF1 Seahawks -69.1 down to DEF31 Titans -189.4), so
    they work as an ordinal 2026 view of each defense."""
    pat = re.compile(r"(?P<name>.+?)\s+(?P<pos>QB|RB|WR|TE|K|DST)\s*\|\s*(?P<nfl>[A-Z]{2,3})")
    by_team = defaultdict(lambda: defaultdict(list))
    dst_proj, dst_rank = {}, {}
    with open(results_csv, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip() or not row[0].strip()[0].isdigit():
                continue
            m = pat.match(row[2].strip().lstrip("*").strip())
            if not m:
                continue
            team = ALIAS.get(m.group("nfl"), m.group("nfl"))
            pts = float(row[5])
            if m.group("pos") == "DST":
                dst_proj[team] = pts
                dst_rank[team] = int(row[4].replace("#", "").split()[0])
            elif m.group("pos") in ("QB", "RB", "WR", "TE"):
                by_team[team][m.group("pos")].append(pts)

    slots = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    out = {}
    for team, pos in by_team.items():
        total = 0.0
        for p, n in slots.items():
            got = sorted(pos.get(p, []), reverse=True)[:n]
            total += sum(got)
        out[team] = total
    return out, dst_proj, dst_rank


def impute_undrafted(dst_proj, dst_rank, form):
    """Put all 32 defenses on the tracker's single 2026 scale.

    The tracker ranked every defense, but only the drafted ones appear in the
    export — so the eleven undrafted teams are exactly the eleven missing DEF
    rank slots (17, 18, 19, 22-27, 30, 32). Order those eleven by last season's
    points allowed, drop them into the missing slots, and read a projection off
    the drafted teams' rank-to-projection curve. This matters: a fallback built
    from 2025 alone rates Cleveland an above-average defense, when the league's
    own board has it 17th and nobody drafted it. Same scale for all 32 beats a
    second, more generous scale for the streaming pool.
    """
    anchors = sorted((dst_rank[t], dst_proj[t]) for t in dst_proj)
    missing = [r for r in range(1, 33) if r not in dst_rank.values()]
    undrafted = sorted((t for t in form if t not in dst_proj), key=lambda t: form[t]["pa"])
    if len(missing) != len(undrafted):
        raise ValueError(f"{len(missing)} open DEF ranks but {len(undrafted)} undrafted teams")

    def interp(rank):
        if rank <= anchors[0][0]:
            lo, hi = anchors[0], anchors[1]
        elif rank >= anchors[-1][0]:
            lo, hi = anchors[-2], anchors[-1]
        else:
            lo = max((a for a in anchors if a[0] <= rank), key=lambda a: a[0])
            hi = min((a for a in anchors if a[0] >= rank), key=lambda a: a[0])
            if lo[0] == hi[0]:
                return lo[1]
        slope = (hi[1] - lo[1]) / (hi[0] - lo[0])
        return lo[1] + slope * (rank - lo[0])

    proj = dict(dst_proj)
    rank = dict(dst_rank)
    imputed = {}
    for team, r in zip(undrafted, missing):
        proj[team] = round(interp(r), 2)
        rank[team] = r
        imputed[team] = r
    return proj, rank, imputed


def z(values):
    m, sd = statistics.mean(values.values()), statistics.pstdev(list(values.values()))
    return {k: (v - m) / sd for k, v in values.items()} if sd else {k: 0.0 for k in values}


# ---------------------------------------------------------------- calibration

def ols(rows, ncol):
    """Least squares via normal equations with Gaussian elimination — small,
    well-conditioned system, no third-party dependency."""
    A = [[0.0] * (ncol + 1) for _ in range(ncol)]
    for x, y in rows:
        for i in range(ncol):
            for j in range(ncol):
                A[i][j] += x[i] * x[j]
            A[i][ncol] += x[i] * y
    for c in range(ncol):
        p = max(range(c, ncol), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        if abs(A[c][c]) < 1e-12:
            raise ValueError("singular design matrix")
        for r in range(ncol):
            if r == c:
                continue
            f = A[r][c] / A[c][c]
            for k in range(c, ncol + 1):
                A[r][k] -= f * A[c][k]
    return [A[i][ncol] / A[i][i] for i in range(ncol)]


def implied_pa(game):
    """Market-implied points for each side: half the total, shifted by half the
    spread. nflverse spread_line is positive when the home team is favoured."""
    if game["total"] is None or game["spread"] is None:
        return None
    home = (game["total"] + game["spread"]) / 2
    away = (game["total"] - game["spread"]) / 2
    return {"home_scores": home, "away_scores": away}


def calibrate(games, off_z, def_z):
    """Fit expected points allowed = b0 + kO*offence(opp) - kD*defense + hfa*home
    against the market lines that exist. Returns coefficients and fit quality."""
    rows, n = [], 0
    for g in games:
        imp = implied_pa(g)
        if not imp:
            continue
        for side, opp, at_home, allowed in (
            (g["home"], g["away"], 1, imp["away_scores"]),
            (g["away"], g["home"], 0, imp["home_scores"]),
        ):
            rows.append(([1.0, off_z.get(opp, 0.0), -def_z.get(side, 0.0), float(at_home)], allowed))
            n += 1
    b0, kO, kD, hfa = ols(rows, 4)
    err = [y - (b[0] * b0 + b[1] * kO + b[2] * kD + b[3] * hfa) for b, y in rows]
    return {"b0": b0, "kO": kO, "kD": kD, "hfa": hfa,
            "n": n, "rmse": math.sqrt(sum(e * e for e in err) / len(err)),
            "mae": sum(abs(e) for e in err) / len(err)}


# ---------------------------------------------------------------- projection

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def tier_points(mu, sd):
    """E[tier points] with points allowed ~ Normal(mu, sd), discretised on the
    tier edges. The tier table is a step function, so this is a weighted sum of
    tier values by the probability mass in each tier."""
    total = 0.0
    for lo, hi, pts in PA_TIERS:
        p = norm_cdf((hi + 0.5 - mu) / sd) - norm_cdf((lo - 0.5 - mu) / sd)
        total += p * pts
    return total


def project(games, weeks, off_z, def_z, coef, sd):
    """Every defense × every week: expected points allowed, tier points, big-play
    points, and the total."""
    cells = defaultdict(dict)
    for g in games:
        for side, opp, at_home in ((g["home"], g["away"], 1), (g["away"], g["home"], 0)):
            pa = (coef["b0"] + coef["kO"] * off_z.get(opp, 0.0)
                  - coef["kD"] * def_z.get(side, 0.0) + coef["hfa"] * at_home)
            tp = tier_points(pa, sd)
            bp = BIGPLAY_BASE + BIGPLAY_D * def_z.get(side, 0.0) + BIGPLAY_O * off_z.get(opp, 0.0)
            imp = implied_pa(g)
            cells[side][g["week"]] = {
                "opp": opp, "home": bool(at_home), "gameday": g["gameday"],
                "exp_pa": round(pa, 1), "tier_pts": round(tp, 2),
                "bigplay_pts": round(bp, 2), "proj": round(tp + bp, 1),
                "market_pa": round(imp["away_scores" if at_home else "home_scores"], 1) if imp else None,
            }
    for team in cells:
        for w in weeks:
            cells[team].setdefault(w, None)  # bye
    return cells


def recommend(cells, weeks, rostered, available):
    """Per week: the best start on the roster, and whether a free agent beats it
    by enough to be worth the transaction."""
    out = {}
    for w in weeks:
        ranked = sorted(((t, cells[t][w]["proj"]) for t in rostered if cells[t].get(w)),
                        key=lambda kv: -kv[1])
        fa = sorted(((t, cells[t][w]["proj"]) for t in available if cells[t].get(w)),
                    key=lambda kv: -kv[1])
        start = ranked[0][0] if ranked else None
        best_fa = fa[0] if fa else None
        upgrade = (best_fa and (not ranked or best_fa[1] - ranked[0][1] >= 1.5))
        out[w] = {
            "start": start,
            "start_proj": ranked[0][1] if ranked else None,
            "bench": [t for t, _ in ranked[1:]],
            "on_bye": [t for t in rostered if not cells[t].get(w)],
            "stream": best_fa[0] if upgrade else None,
            "stream_proj": best_fa[1] if upgrade else None,
            "stream_gain": round(best_fa[1] - ranked[0][1], 1) if upgrade and ranked else None,
        }
    return out


def current_week(games, today):
    """The week whose games are still mostly ahead of us."""
    for w in sorted({g["week"] for g in games}):
        last = max(g["gameday"] for g in games if g["week"] == w)
        if last >= today:
            return w
    return max(g["week"] for g in games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-pull schedule and prior-season form")
    ap.add_argument("--results", default="hh_draft_2026.csv")
    ap.add_argument("--rostered", default="TB,NYG,LV")
    ap.add_argument("--focus", default="KGB")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD; defaults to today")
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args()

    if args.refresh or not (os.path.exists(SCHEDULE_CSV) and os.path.exists(FORM_CSV)):
        refresh()

    games, weeks = load_schedule()
    form, resid_sd = load_form()
    off_proxy, dst_proj_drafted, dst_rank_drafted = draft_offence(args.results)
    dst_proj, dst_rank, imputed = impute_undrafted(dst_proj_drafted, dst_rank_drafted, form)

    # offence: this league's own projections, anchored by last season's scoring
    off_draft_z = z(off_proxy)
    off_form_z = z({t: v["pf"] for t, v in form.items()})
    off_z = {t: 0.55 * off_draft_z.get(t, 0.0) + 0.45 * off_form_z.get(t, 0.0) for t in form}

    # defense: the tracker's 2026 DST ranking, anchored by last season's points
    # allowed. Same 55/45 blend and same scale for all 32 defenses.
    def_form_z = z({t: -v["pa"] for t, v in form.items()})
    def_proj_z = z(dst_proj)
    def_z = {t: 0.55 * def_proj_z[t] + 0.45 * def_form_z[t] for t in form}

    coef = calibrate(games, off_z, def_z)
    cells = project(games, weeks, off_z, def_z, coef, resid_sd)

    rostered = [ALIAS.get(t.strip().upper(), t.strip().upper()) for t in args.rostered.split(",")]
    drafted = set(dst_proj_drafted)
    available = sorted(t for t in cells if t not in drafted and t not in rostered)

    today = args.today or __import__("datetime").date.today().isoformat()
    recs = recommend(cells, weeks, rostered, available)

    season_total = {t: round(sum(c["proj"] for c in cells[t].values() if c), 1) for t in cells}
    out = {
        "meta": {
            "focus": args.focus,
            "season": 2026,
            "weeks": weeks,
            "today": today,
            "current_week": current_week(games, today),
            "rostered": rostered,
            "available": available,
            "drafted_elsewhere": sorted(drafted - set(rostered)),
            "team_name": TEAM_NAME,
            "pa_tiers": [{"lo": lo, "hi": hi, "pts": p} for lo, hi, p in PA_TIERS],
            "coef": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in coef.items()},
            "residual_sd": resid_sd,
            "bigplay": {"base": BIGPLAY_BASE, "per_sd_def": BIGPLAY_D, "per_sd_off": BIGPLAY_O},
            "index": {t: {"off": round(off_z[t], 2), "def": round(def_z[t], 2),
                          "def_rank": dst_rank[t], "def_proj": dst_proj[t],
                          "pa_pg_2025": form[t]["pa"], "pf_pg_2025": form[t]["pf"],
                          "drafted": t in drafted, "rank_imputed": t in imputed}
                      for t in sorted(cells)},
        },
        "cells": {t: {str(w): c for w, c in sorted(ws.items())} for t, ws in cells.items()},
        "season_total": season_total,
        "recommend": {str(w): r for w, r in recs.items()},
    }

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "dst.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"wrote {args.outdir}/dst.json")
    print(f"  calibration on {coef['n']} market team-games: "
          f"kO {coef['kO']:.2f} kD {coef['kD']:.2f} hfa {coef['hfa']:.2f} "
          f"(MAE {coef['mae']:.2f} pts, RMSE {coef['rmse']:.2f})")
    print(f"  points-allowed sigma {resid_sd:.2f}")
    best = sorted(season_total.items(), key=lambda kv: -kv[1])[:5]
    print("  best full-season defenses: " + ", ".join(f"{t} {v}" for t, v in best))
    for w in weeks[:6]:
        r = recs[w]
        s = f"  wk{w}: start {r['start']} ({r['start_proj']})"
        if r["stream"]:
            s += f" — but {r['stream']} projects {r['stream_proj']} (+{r['stream_gain']}) off waivers"
        print(s)


if __name__ == "__main__":
    raise SystemExit(main())
