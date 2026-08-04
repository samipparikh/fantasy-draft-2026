#!/usr/bin/env python3
"""
draft_model.py -- snake-draft model for a 12-team, half-PPR league, drafting from
slots 6 and 7, with an RB-priority bias.

What it does
------------
1. Loads a player pool (name, pos, team, adp, proj_pts) from CSV. Half-PPR points
   are computed from projected stats if proj_pts is missing.
2. Computes VOR (value over replacement) with dynamic, roster-aware baselines,
   and detects tier cliffs inside each position.
3. Monte-Carlo simulates the other 11 teams using ADP + noise, producing
   P(player still available) at each of *your* picks.
4. Runs a strategy tournament (rb_priority / bpa / hero_rb / zero_rb) for each
   draft slot, scoring each simulated roster by its projected starting lineup.
5. Emits a round-by-round plan: position mix by pick, top realistic targets with
   availability odds, and where the positional cliffs land relative to your picks.

Usage
-----
    python3 draft_model.py --players players.csv
    python3 draft_model.py --players players.csv --sims 4000 --slots 6 7
    python3 draft_model.py --make-template players.csv   # write a blank CSV
    python3 draft_model.py --demo                        # synthetic pool, smoke test

The --demo pool is FAKE data for verifying the engine only. Real conclusions
require your own ADP/projection export.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

POS_ALIASES = {
    "D": "DST", "DEF": "DST", "DST": "DST", "D/ST": "DST", "DEFENSE": "DST",
    "PK": "K", "K": "K",
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "FB": "RB",
}

# Half-PPR scoring, used only when proj_pts is absent and raw stats are present.
HALF_PPR = {
    "pass_yds": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yds": 0.10, "rush_td": 6.0,
    "rec": 0.5, "rec_yds": 0.10, "rec_td": 6.0,
    "fum_lost": -2.0, "two_pt": 2.0,
}

# Share of the single FLEX slot each position is expected to win, league-wide.
# Drives the replacement-level baseline. Half PPR with 2WR is RB-friendly.
FLEX_SHARE = {"RB": 0.45, "WR": 0.50, "TE": 0.05}

# How many of a position a realistic opponent will roster, and when.
OPP_POS_MAX = {"QB": 2, "RB": 6, "WR": 6, "TE": 2, "K": 1, "DST": 1}
OPP_EARLIEST_ROUND = {"K": -2, "DST": -2}  # negative => counts back from the end
OPP_SECOND_ALLOWED_ROUND = {"QB": 8, "TE": 9}


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #

class Player:
    __slots__ = ("name", "pos", "team", "adp", "pts", "vor", "tier", "idx", "bye", "av")

    def __init__(self, name, pos, team, adp, pts, bye="", av=0.0):
        self.name = name
        self.pos = pos
        self.team = team
        self.adp = adp
        self.pts = pts
        self.bye = bye
        self.av = av
        self.vor = 0.0
        self.tier = 0
        self.idx = -1

    def __repr__(self):
        return f"<{self.name} {self.pos} adp={self.adp:.1f} vor={self.vor:.1f} T{self.tier}>"

    @property
    def label(self):
        return f"{self.name} ({self.pos}{self.tier}"  + (f", {self.team})" if self.team else ")")


class League:
    def __init__(self, teams, rounds, starters, flex, sigma_frac, tier_gap):
        self.teams = teams
        self.rounds = rounds
        self.starters = starters          # {pos: count}, excludes FLEX
        self.flex = flex                  # number of flex (RB/WR/TE) slots
        self.sigma_frac = sigma_frac
        self.tier_gap = tier_gap

    @property
    def total_picks(self):
        return self.teams * self.rounds

    def pick_numbers(self, slot):
        """Overall pick numbers (1-indexed) for a given draft slot in a snake."""
        out = []
        for r in range(1, self.rounds + 1):
            if r % 2 == 1:
                out.append((r - 1) * self.teams + slot)
            else:
                out.append(r * self.teams - slot + 1)
        return out

    def team_on_the_clock(self, overall_pick):
        r = (overall_pick - 1) // self.teams + 1
        i = (overall_pick - 1) % self.teams + 1
        return i if r % 2 == 1 else self.teams - i + 1

    def round_of(self, overall_pick):
        return (overall_pick - 1) // self.teams + 1


# --------------------------------------------------------------------------- #
# loading + valuation
# --------------------------------------------------------------------------- #

def norm_pos(raw):
    key = (raw or "").strip().upper().replace(" ", "")
    return POS_ALIASES.get(key)


def to_float(row, key):
    v = (row.get(key) or "").strip().replace(",", "")
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_players(path):
    """Read the player CSV. Tolerant of extra columns and header casing."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path}: no rows found")

    rows = [{(k or "").strip().lower(): v for k, v in r.items()} for r in rows]
    players, skipped, derived_pts = [], [], 0

    for r in rows:
        name = (r.get("name") or r.get("player") or "").strip()
        pos = norm_pos(r.get("pos") or r.get("position"))
        if not name or not pos:
            skipped.append(r)
            continue
        adp = to_float(r, "adp") or to_float(r, "rank") or to_float(r, "ecr")
        pts = to_float(r, "proj_pts") or to_float(r, "points") or to_float(r, "fpts")
        if pts is None:
            stat_pts = sum(HALF_PPR[k] * (to_float(r, k) or 0.0) for k in HALF_PPR)
            if stat_pts > 0:
                pts, derived_pts = stat_pts, derived_pts + 1
        if adp is None and pts is None:
            skipped.append(r)
            continue
        av = to_float(r, "av") or to_float(r, "auction") or to_float(r, "value") or 0.0
        players.append(Player(name, pos, (r.get("team") or "").strip().upper(),
                              adp if adp is not None else 999.0, pts,
                              (r.get("bye") or "").strip(), av))

    # Fill missing ADP from projection rank, and vice versa, so the pool is usable.
    fill_missing_adp(players)
    n_pts = sum(1 for p in players if p.pts is not None)
    n_av = sum(1 for p in players if p.av)
    # Prefer real projections; else auction value; else the ADP decay curve.
    if n_pts < 0.5 * len(players) and n_av >= 0.15 * len(players):
        apply_av_values(players)
        value_src = f"auction value ({n_av} priced players)"
    else:
        fill_missing_pts(players)
        value_src = "projected points" if n_pts else "ADP decay curve (no projections)"

    players.sort(key=lambda p: p.adp)
    for i, p in enumerate(players):
        p.idx = i
    return players, skipped, derived_pts, value_src


def fill_missing_adp(players):
    known = [p for p in players if p.adp < 999.0]
    missing = [p for p in players if p.adp >= 999.0]
    if not missing:
        return
    if not known:
        for i, p in enumerate(sorted(players, key=lambda x: -(x.pts or 0.0)), start=1):
            p.adp = float(i)
        return
    floor = max(p.adp for p in known)
    for i, p in enumerate(sorted(missing, key=lambda x: -(x.pts or 0.0)), start=1):
        p.adp = floor + i


# Plausible half-PPR replacement-level points for a 12-team league. Used only to
# put AV-derived values on a points-like scale; VOR is unaffected by the offsets.
AV_REPLACEMENT = {"QB": 250.0, "RB": 105.0, "WR": 110.0, "TE": 85.0,
                  "K": 115.0, "DST": 105.0}
# Dollars -> season points. Anchored so the $65 overall AV leader lands near a
# 225-point VOR, matching a typical RB1 half-PPR season over replacement.
AV_TO_PTS = 3.5


def apply_av_values(players):
    """Derive points from auction value.

    Auction dollars are themselves a value-over-replacement calculation, so
    AV -> VOR is a linear transform. This keeps the ranker's positional scarcity
    judgement instead of re-deriving it from raw ADP order.

    Players at $0 carry no dollar resolution, so they are ordered by ADP just
    below replacement level.
    """
    zero = [p for p in players if not p.av]
    zero.sort(key=lambda p: p.adp)
    zero_rank = {id(p): i for i, p in enumerate(zero)}
    for p in players:
        base = AV_REPLACEMENT.get(p.pos, 110.0)
        if p.av:
            p.pts = base + AV_TO_PTS * p.av
        else:
            # monotone, slightly below replacement, preserving ADP order
            p.pts = base - 1.0 - 0.25 * zero_rank.get(id(p), 0)


def fill_missing_pts(players):
    """If projections are absent, fit a per-position decay of points vs ADP."""
    have = [p for p in players if p.pts is not None]
    if len(have) >= max(24, 0.5 * len(players)):
        # Enough real projections: patch the stragglers positionally.
        by_pos = defaultdict(list)
        for p in have:
            by_pos[p.pos].append(p)
        for p in players:
            if p.pts is None:
                pool = sorted(by_pos.get(p.pos, have), key=lambda x: abs(x.adp - p.adp))[:5]
                p.pts = statistics.fmean(x.pts for x in pool) if pool else 0.0
        return
    # No usable projections at all: synthesize a monotone value curve from ADP.
    # Shape only -- absolute points are meaningless, but VOR ordering survives.
    anchor = {"QB": 320.0, "RB": 300.0, "WR": 290.0, "TE": 210.0, "K": 130.0, "DST": 130.0}
    decay = {"QB": 0.006, "RB": 0.011, "WR": 0.009, "TE": 0.012, "K": 0.004, "DST": 0.004}
    rank_in_pos = defaultdict(int)
    for p in sorted(players, key=lambda x: x.adp):
        rank_in_pos[p.pos] += 1
        n = rank_in_pos[p.pos]
        p.pts = anchor.get(p.pos, 200.0) * math.exp(-decay.get(p.pos, 0.01) * (n - 1) * 3.2)


def baselines(players, lg):
    """Replacement level per position: mean points of the 3 players straddling the
    last projected league-wide starter at that position (starters + flex share)."""
    out = {}
    for pos in POSITIONS:
        pool = sorted([p for p in players if p.pos == pos], key=lambda p: -p.pts)
        if not pool:
            continue
        n_start = lg.starters.get(pos, 0) + lg.flex * FLEX_SHARE.get(pos, 0.0)
        rank = max(1, int(round(lg.teams * n_start)))
        lo, hi = max(0, rank - 2), min(len(pool), rank + 2)
        window = pool[lo:hi] or pool[-1:]
        out[pos] = (statistics.fmean(p.pts for p in window), rank)
    return out


def compute_vor(players, lg):
    base = baselines(players, lg)
    for p in players:
        repl = base.get(p.pos, (0.0, 0))[0]
        p.vor = p.pts - repl
    return base


def assign_tiers(players, lg):
    """Tier break on a VOR gap to the next player at the position.

    The absolute-gap threshold alone lumps the whole mid/late board into one
    giant tier, because VOR gaps shrink as you go down a position. So the
    threshold decays with depth: a break near the top needs a big gap, a break
    100 picks deep needs much less. Tiers are also capped in size so a long flat
    stretch still gets subdivided into usable chunks.
    """
    for pos in POSITIONS:
        pool = sorted([p for p in players if p.pos == pos], key=lambda p: -p.vor)
        if not pool:
            continue
        span = max(1.0, pool[0].vor - pool[-1].vor)
        tier, in_tier = 1, 0
        for i, p in enumerate(pool):
            if i > 0:
                gap = pool[i - 1].vor - p.vor
                # required gap decays from tier_gap toward ~15% of it with depth
                need = lg.tier_gap * max(0.15, 1.0 - 0.055 * (i - 1))
                # and is never more than a fifth of the position's whole spread
                need = min(need, 0.20 * span)
                if gap >= need or in_tier >= 8:
                    tier += 1
                    in_tier = 0
            p.tier = tier
            in_tier += 1


def tier_cliffs(players, lg, pos, max_tier=5):
    """Where each tier at a position runs out, in ADP terms."""
    pool = sorted([p for p in players if p.pos == pos], key=lambda p: -p.vor)
    out = []
    for t in range(1, max_tier + 1):
        members = [p for p in pool if p.tier == t]
        if not members:
            continue
        out.append({
            "tier": t,
            "count": len(members),
            "vor_hi": members[0].vor,
            "vor_lo": members[-1].vor,
            "last_adp": max(p.adp for p in members),
            "names": [p.name for p in members],
        })
    return out


# --------------------------------------------------------------------------- #
# opponent behaviour
# --------------------------------------------------------------------------- #

def noisy_order(players, lg, rng):
    """One simulated 'consensus board': ADP perturbed by noise that grows with ADP."""
    scored = []
    for p in players:
        sigma = max(2.5, lg.sigma_frac * p.adp)
        scored.append((p.adp + rng.gauss(0.0, sigma), p.idx))
    scored.sort()
    return [i for _, i in scored]


def opp_eligible(pos, roster_counts, rnd, lg):
    cap = OPP_POS_MAX.get(pos, 8)
    have = roster_counts.get(pos, 0)
    if have >= cap:
        return False
    earliest = OPP_EARLIEST_ROUND.get(pos)
    if earliest is not None:
        first_ok = lg.rounds + earliest + 1 if earliest < 0 else earliest
        if rnd < first_ok:
            return False
    if have >= 1 and pos in OPP_SECOND_ALLOWED_ROUND and rnd < OPP_SECOND_ALLOWED_ROUND[pos]:
        return False
    return True


# --------------------------------------------------------------------------- #
# our strategies
# --------------------------------------------------------------------------- #

STRATEGIES = ("rb_priority", "bpa", "hero_rb", "zero_rb")

STRATEGY_DOC = {
    "rb_priority": "RB-weighted VONA. Pushes RB early and often without ignoring value.",
    "bpa":         "Pure value over replacement + VONA. No positional thumb on the scale.",
    "hero_rb":     "One elite RB in round 1, then WR/TE until round 5+.",
    "zero_rb":     "No RB before round 4; load WR/TE early, attack RB volume late.",
}


def pos_multiplier(strategy, pos, rnd, rb_weight):
    if pos in ("K", "DST"):
        return 1.0
    if strategy == "rb_priority":
        if pos == "RB":
            return rb_weight if rnd <= 6 else 1.0 + (rb_weight - 1.0) * 0.5
        return 1.0
    if strategy == "hero_rb":
        if pos == "RB":
            return 1.35 if rnd == 1 else (0.55 if rnd <= 4 else 1.05)
        return 1.0
    if strategy == "zero_rb":
        if pos == "RB":
            return 0.35 if rnd <= 3 else (1.15 if rnd >= 6 else 1.0)
        return 1.10 if pos in ("WR", "TE") and rnd <= 4 else 1.0
    return 1.0


def our_limits(lg):
    """Max we will roster at each position over the whole draft."""
    return {"QB": 2, "RB": 7, "WR": 7, "TE": 2,
            "K": lg.starters.get("K", 1), "DST": lg.starters.get("DST", 1)}


def required_remaining(counts, lg):
    """Starter slots we still must fill (flex counted as one generic RB/WR/TE)."""
    need = 0
    for pos, n in lg.starters.items():
        need += max(0, n - counts.get(pos, 0))
    flexable = sum(max(0, counts.get(p, 0) - lg.starters.get(p, 0)) for p in ("RB", "WR", "TE"))
    need += max(0, lg.flex - flexable)
    return need


def need_multiplier(pos, counts, rnd, lg):
    """Urgency: unfilled starter slots matter more as the draft runs out."""
    if pos in ("K", "DST"):
        # Only worth anything in the final rounds, then mandatory.
        if rnd >= lg.rounds - 1 and counts.get(pos, 0) < lg.starters.get(pos, 0):
            return 6.0
        return 0.0
    unfilled = max(0, lg.starters.get(pos, 0) - counts.get(pos, 0))
    m = 1.0 + 0.22 * unfilled
    # Backup QB/TE have little value until the roster is otherwise set.
    if pos in ("QB", "TE") and counts.get(pos, 0) >= lg.starters.get(pos, 0):
        m *= 0.35 if rnd < lg.rounds - 4 else 0.7
    rounds_left = lg.rounds - rnd + 1
    if required_remaining(counts, lg) >= rounds_left and unfilled > 0:
        m *= 3.0  # forced to fill
    return m


def expected_best_vor(cands, prob):
    """E[max VOR] over candidates given independent availability probabilities.
    cands must be sorted by descending VOR. Anything that survives to 0 is
    treated as replacement level (VOR 0)."""
    e, carry = 0.0, 1.0
    for p in cands:
        pr = prob(p)
        if pr <= 0.0:
            continue
        e += carry * pr * p.vor
        carry *= (1.0 - pr)
        if carry < 1e-3:
            break
    return e


def choose_pick(strategy, avail_by_pos, counts, rnd, lg, cfg, surv_ratio):
    """Score every plausible candidate and return the best.

    surv_ratio(pos_player) -> P(this player lasts until our next pick | here now)
    """
    limits = our_limits(lg)
    best, best_score = None, -1e18
    for pos, pool in avail_by_pos.items():
        if not pool:
            continue
        if counts.get(pos, 0) >= limits.get(pos, 8):
            continue
        nm = need_multiplier(pos, counts, rnd, lg)
        if nm <= 0.0:
            continue
        pm = pos_multiplier(strategy, pos, rnd, cfg.rb_weight)
        # Only the top few at a position can ever win; scanning all is wasted work.
        head = pool[: cfg.candidates_per_pos]
        e_next = expected_best_vor(pool[: cfg.vona_depth], surv_ratio) if surv_ratio else 0.0
        for p in head:
            vona = p.vor - e_next
            score = (p.vor + cfg.vona_weight * vona) * pm * nm
            if score > best_score:
                best, best_score = p, score
    if best is None:  # roster caps exhausted the board; fall back to raw VOR
        for pool in avail_by_pos.values():
            if pool and (best is None or pool[0].vor > best.vor):
                best = pool[0]
    return best


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #

class SimResult:
    def __init__(self, lg, my_picks):
        self.n = 0
        self.avail_counts = [Counter() for _ in my_picks]   # player idx -> times available
        self.pos_taken = [Counter() for _ in my_picks]      # pos -> times we took it
        self.player_taken = [Counter() for _ in my_picks]
        self.lineup_pts = []
        self.roster_pos = Counter()


def starting_lineup_points(roster, lg):
    """Best legal starting lineup from a roster, by projected points."""
    by_pos = defaultdict(list)
    for p in roster:
        by_pos[p.pos].append(p)
    for pool in by_pos.values():
        pool.sort(key=lambda p: -p.pts)

    total, used = 0.0, set()
    for pos, n in lg.starters.items():
        for p in by_pos.get(pos, [])[:n]:
            total += p.pts
            used.add(id(p))
    flex_pool = sorted(
        [p for p in roster if p.pos in ("RB", "WR", "TE") and id(p) not in used],
        key=lambda p: -p.pts,
    )
    for p in flex_pool[: lg.flex]:
        total += p.pts
    return total


def run_sims(players, lg, cfg, slot, strategy, surv=None, seed=0):
    """Simulate `cfg.sims` drafts. If `surv` is given (a list of dicts mapping
    player idx -> P(available) at each of our picks), our picks use VONA."""
    rng = random.Random(seed)
    my_picks = lg.pick_numbers(slot)
    my_pick_set = {pk: i for i, pk in enumerate(my_picks)}
    res = SimResult(lg, my_picks)
    n_players = len(players)

    # P(survive from our pick i to our pick i+1 | available at i), per player.
    ratio = None
    if surv:
        ratio = []
        for i in range(len(my_picks)):
            d = {}
            if i + 1 < len(surv):
                for idx, pa in surv[i].items():
                    if pa > 0.02:
                        d[idx] = min(1.0, surv[i + 1].get(idx, 0.0) / pa)
            ratio.append(d)

    for s in range(cfg.sims):
        order = noisy_order(players, lg, rng)
        taken = bytearray(n_players)
        rosters = defaultdict(Counter)
        my_roster = []
        head = 0  # first index in `order` that is still untaken

        # Our available pool, kept per position and sorted by VOR (lazy cleanup).
        avail_by_pos = {pos: sorted([p for p in players if p.pos == pos],
                                    key=lambda p: -p.vor)
                        for pos in POSITIONS}

        for overall in range(1, lg.total_picks + 1):
            rnd = lg.round_of(overall)
            team = lg.team_on_the_clock(overall)
            while head < n_players and taken[order[head]]:
                head += 1

            if team == slot:
                pi = my_pick_set[overall]
                for pos in avail_by_pos:
                    pool = avail_by_pos[pos]
                    while pool and taken[pool[0].idx]:
                        pool.pop(0)
                    avail_by_pos[pos] = [p for p in pool if not taken[p.idx]] \
                        if len(pool) and s == -1 else pool
                # record availability of the meaningful part of the board
                if cfg.track_avail:
                    c = res.avail_counts[pi]
                    for pos, pool in avail_by_pos.items():
                        for p in pool[: cfg.track_top]:
                            if not taken[p.idx]:
                                c[p.idx] += 1

                sr = None
                if ratio is not None and pi < len(ratio) and ratio[pi]:
                    rmap = ratio[pi]
                    sr = lambda p, _r=rmap: _r.get(p.idx, 0.0)

                clean = {pos: [p for p in pool if not taken[p.idx]]
                         for pos, pool in avail_by_pos.items()}
                pick = choose_pick(strategy, clean, rosters[slot], rnd, lg, cfg, sr)
                if pick is None:
                    continue
                taken[pick.idx] = 1
                rosters[slot][pick.pos] += 1
                my_roster.append(pick)
                res.pos_taken[pi][pick.pos] += 1
                res.player_taken[pi][pick.idx] += 1
                avail_by_pos = clean
                avail_by_pos[pick.pos] = [p for p in avail_by_pos[pick.pos] if p.idx != pick.idx]
            else:
                j = head
                chosen = None
                while j < n_players:
                    idx = order[j]
                    if not taken[idx]:
                        p = players[idx]
                        if opp_eligible(p.pos, rosters[team], rnd, lg):
                            chosen = p
                            break
                    j += 1
                if chosen is None:
                    j = head
                    while j < n_players and taken[order[j]]:
                        j += 1
                    if j >= n_players:
                        break
                    chosen = players[order[j]]
                taken[chosen.idx] = 1
                rosters[team][chosen.pos] += 1

        res.n += 1
        res.lineup_pts.append(starting_lineup_points(my_roster, lg))
        for p in my_roster:
            res.roster_pos[p.pos] += 1

    return res, my_picks


def survival_table(res, my_picks):
    return [{idx: c / max(1, res.n) for idx, c in counter.items()}
            for counter in res.avail_counts]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def pct(x):
    return f"{100.0 * x:4.0f}%"


def fmt_table(headers, rows, aligns=None):
    cols = len(headers)
    widths = [len(h) for h in headers]
    srows = []
    for r in rows:
        cells = [("" if c is None else str(c)) for c in r]
        srows.append(cells)
        for i, c in enumerate(cells[:cols]):
            widths[i] = max(widths[i], len(c))
    aligns = aligns or ["<"] * cols
    out = ["  ".join(f"{h:{aligns[i]}{widths[i]}}" for i, h in enumerate(headers)),
           "  ".join("-" * widths[i] for i in range(cols))]
    for cells in srows:
        out.append("  ".join(f"{cells[i]:{aligns[i]}{widths[i]}}" for i in range(cols)))
    return "\n".join(out)


def report_header(lg, players, base, skipped, derived, cfg, lines, value_src=""):
    starters = ", ".join(f"{k}{v}" for k, v in lg.starters.items()) + f", FLEX{lg.flex}"
    lines.append("=" * 78)
    lines.append("HALF-PPR SNAKE DRAFT MODEL")
    lines.append("=" * 78)
    lines.append(f"League      : {lg.teams} teams, {lg.rounds} rounds, snake")
    lines.append(f"Starters    : {starters}")
    lines.append(f"Scoring     : half PPR (0.5/rec)")
    if value_src:
        lines.append(f"Value from  : {value_src}")
    lines.append(f"Pool        : {len(players)} players"
                 + (f", {len(skipped)} rows skipped (no pos/name/value)" if skipped else "")
                 + (f", {derived} projections computed from raw stats" if derived else ""))
    lines.append(f"Sims        : {cfg.sims} per slot per strategy, ADP noise sigma = "
                 f"{lg.sigma_frac:.2f} x ADP (floor 2.5)")
    lines.append("")
    lines.append("Replacement baselines (VOR = projected points - baseline):")
    rows = [[pos, f"{pos}{r}", f"{v:.1f} pts"] for pos, (v, r) in
            sorted(base.items(), key=lambda kv: POSITIONS.index(kv[0]))]
    lines.append(fmt_table(["POS", "REPLACEMENT", "BASELINE"], rows))
    lines.append("")


def report_cliffs(players, lg, lines):
    lines.append("-" * 78)
    lines.append(f"TIER CLIFFS  (break on a VOR gap, threshold {lg.tier_gap:.0f} pts near the")
    lines.append("top of a position and decaying with depth; max 8 per tier)")
    lines.append("-" * 78)
    for pos in ("RB", "WR", "TE", "QB"):
        cliffs = tier_cliffs(players, lg, pos, max_tier=4)
        if not cliffs:
            continue
        rows = []
        for c in cliffs:
            preview = ", ".join(c["names"][:4]) + (" ..." if len(c["names"]) > 4 else "")
            rows.append([f"{pos}{c['tier']}", c["count"],
                         f"{c['vor_hi']:.0f}..{c['vor_lo']:.0f}",
                         f"{c['last_adp']:.0f}", preview])
        lines.append(fmt_table([f"{pos} TIER", "N", "VOR RANGE", "LAST ADP", "MEMBERS"],
                               rows, ["<", ">", ">", ">", "<"]))
        lines.append("")


def report_strategies(slot, results, lines):
    lines.append("-" * 78)
    lines.append(f"STRATEGY TOURNAMENT -- SLOT {slot}")
    lines.append("-" * 78)
    lines.append("Score = projected points of the best legal starting lineup the strategy")
    lines.append("ended up with, across simulated drafts.")
    lines.append("")
    rows = []
    for name, res in results.items():
        pts = sorted(res.lineup_pts)
        mean = statistics.fmean(pts)
        rows.append([name, f"{mean:.1f}",
                     f"{pts[int(0.10 * (len(pts) - 1))]:.0f}",
                     f"{pts[len(pts) // 2]:.0f}",
                     f"{pts[int(0.90 * (len(pts) - 1))]:.0f}",
                     STRATEGY_DOC[name]])
    rows.sort(key=lambda r: -float(r[1]))
    best = rows[0][0]
    for r in rows:
        r[0] = ("* " if r[0] == best else "  ") + r[0]
    lines.append(fmt_table(["STRATEGY", "MEAN", "P10", "P50", "P90", "NOTES"], rows,
                           ["<", ">", ">", ">", ">", "<"]))
    lines.append("")
    delta = float(rows[0][1]) - float(rows[-1][1])
    lines.append(f"Spread between best and worst strategy: {delta:.1f} projected points "
                 f"({delta / 17.0:.1f} pts/week).")
    lines.append("")
    return best


def report_plan(slot, my_picks, res, surv, players, lg, cfg, lines, strat_name=""):
    lines.append("-" * 78)
    lines.append(f"ROUND-BY-ROUND PLAN -- SLOT {slot}"
                 + (f"  (strategy: {strat_name})" if strat_name else ""))
    lines.append("-" * 78)
    lines.append("POSITION MIX = how often this strategy took that position here.")
    lines.append("TARGETS = players most often drafted at this pick, with P(available).")
    lines.append("")
    for i, pk in enumerate(my_picks):
        rnd = lg.round_of(pk)
        if rnd > cfg.plan_rounds:
            break
        mix = res.pos_taken[i]
        tot = max(1, sum(mix.values()))
        mix_s = "  ".join(f"{pos} {100 * n // tot}%" for pos, n in mix.most_common(4))
        lines.append(f"[Round {rnd:>2}  pick {rnd}.{(pk - 1) % lg.teams + 1:02d}  overall {pk:>3}]  {mix_s}")
        rows = []
        for idx, n in res.player_taken[i].most_common(6):
            p = players[idx]
            pa = surv[i].get(idx, 0.0)
            rows.append([p.name, f"{p.pos}{p.tier}", p.team, f"{p.adp:.0f}",
                         f"{p.vor:.0f}", pct(pa), f"{100 * n // tot}%"])
        if rows:
            lines.append(fmt_table(["TARGET", "POS", "TM", "ADP", "VOR", "P(AVAIL)", "P(TAKE)"],
                                   rows, ["<", "<", "<", ">", ">", ">", ">"]))
        # who is realistically on the board that we might miss
        board = sorted(((surv[i].get(p.idx, 0.0), p) for p in players
                        if surv[i].get(p.idx, 0.0) >= 0.25 and p.pos in ("RB", "WR", "TE", "QB")),
                       key=lambda t: -t[1].vor)[:5]
        if board:
            lines.append("   board: " + " | ".join(
                f"{p.name} {p.pos}{p.tier} {pct(pa).strip()}" for pa, p in board))
        lines.append("")
    return lines


def report_rb_check(slot, results, best, lines):
    """Is the RB preference costing anything on this data?"""
    rb = statistics.fmean(results["rb_priority"].lineup_pts)
    bpa = statistics.fmean(results["bpa"].lineup_pts)
    diff = rb - bpa
    lines.append("-" * 78)
    lines.append(f"RB-PRIORITY SANITY CHECK -- SLOT {slot}")
    lines.append("-" * 78)
    if diff >= 0:
        lines.append(f"RB priority beats pure value by {diff:.1f} projected points. Your")
        lines.append("preference is free on this data -- the RB scarcity is real here.")
    else:
        lines.append(f"RB priority costs {-diff:.1f} projected points vs pure value "
                     f"({-diff / 17.0:.2f}/week).")
        lines.append("That is the price of the preference. Cheap if small; if it is more than")
        lines.append("~15 points, consider lowering --rb-weight or taking value when it falls.")
    avg_rb = results["rb_priority"].roster_pos["RB"] / max(1, results["rb_priority"].n)
    avg_wr = results["rb_priority"].roster_pos["WR"] / max(1, results["rb_priority"].n)
    lines.append(f"Typical rb_priority roster shape: {avg_rb:.1f} RB, {avg_wr:.1f} WR.")
    lines.append(f"Best strategy on this pool at slot {slot}: {best}.")
    lines.append("")


def write_availability_csv(path, my_picks, surv, players, lg):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "pos", "tier", "team", "adp", "proj_pts", "vor"]
                   + [f"p_avail_r{lg.round_of(pk)}" for pk in my_picks])
        seen = set()
        for i in range(len(my_picks)):
            seen |= set(surv[i].keys())
        for idx in sorted(seen, key=lambda i: -players[i].vor):
            p = players[idx]
            w.writerow([p.name, p.pos, p.tier, p.team, f"{p.adp:.1f}",
                        f"{p.pts:.1f}", f"{p.vor:.1f}"]
                       + [f"{surv[i].get(idx, 0.0):.3f}" for i in range(len(my_picks))])


def write_board_csv(path, players):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["vor_rank", "name", "pos", "tier", "team", "bye", "adp",
                    "av", "value_pts", "vor"])
        for i, p in enumerate(sorted(players, key=lambda x: -x.vor), start=1):
            w.writerow([i, p.name, p.pos, p.tier, p.team, p.bye, f"{p.adp:.1f}",
                        f"{p.av:.0f}", f"{p.pts:.1f}", f"{p.vor:.1f}"])


# --------------------------------------------------------------------------- #
# scaffolding
# --------------------------------------------------------------------------- #

TEMPLATE_HEADER = ["name", "pos", "team", "bye", "adp", "proj_pts", "av",
                   "pass_yds", "pass_td", "pass_int", "rush_yds", "rush_td",
                   "rec", "rec_yds", "rec_td", "fum_lost"]


def make_template(path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(TEMPLATE_HEADER)
        w.writerow(["Example Player", "RB", "ATL", "5", "2.1", "278.0", "63",
                    "", "", "", "1250", "12", "58", "480", "3", "2"])
    print(f"wrote {path}")
    print("Required: name, pos, adp. Strongly recommended: proj_pts (half PPR).")
    print("If proj_pts is blank, raw stat columns are scored at half PPR.")
    print("Extra columns are ignored, so a raw FantasyPros/Sleeper export usually works.")


def demo_pool():
    """Synthetic, clearly-fake pool for smoke-testing the engine only."""
    rng = random.Random(7)
    counts = {"QB": 32, "RB": 60, "WR": 75, "TE": 28, "K": 20, "DST": 20}
    top = {"QB": 300.0, "RB": 285.0, "WR": 275.0, "TE": 205.0, "K": 135.0, "DST": 130.0}
    dec = {"QB": 0.030, "RB": 0.038, "WR": 0.030, "TE": 0.045, "K": 0.012, "DST": 0.014}
    players = []
    for pos, n in counts.items():
        for i in range(1, n + 1):
            pts = top[pos] * math.exp(-dec[pos] * (i - 1)) + rng.gauss(0, 6)
            players.append(Player(f"Demo{pos}{i}", pos, "XXX", 0.0, max(20.0, pts)))
    # ADP roughly tracks value, with position-appropriate market bias.
    bias = {"QB": 1.35, "RB": 0.92, "WR": 1.0, "TE": 1.25, "K": 3.0, "DST": 3.0}
    ranked = sorted(players, key=lambda p: -(p.pts * (1.0 / bias[p.pos])))
    for i, p in enumerate(ranked, start=1):
        p.adp = max(1.0, i + rng.gauss(0, 3))
    players.sort(key=lambda p: p.adp)
    for i, p in enumerate(players):
        p.idx = i
    return players


class Cfg:
    pass


def parse_starters(s):
    out, flex = {}, 0
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        k = k.strip().upper()
        n = int(v or 1)
        if k == "FLEX":
            flex = n
        else:
            pos = norm_pos(k)
            if not pos:
                sys.exit(f"unknown position in --starters: {k}")
            out[pos] = n
    return out, flex


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", help="player CSV (name,pos,team,adp,proj_pts,...)")
    ap.add_argument("--make-template", metavar="PATH", help="write a blank player CSV and exit")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic FAKE pool")
    ap.add_argument("--slots", type=int, nargs="+", default=[6, 7], help="draft slots to model")
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--starters", default="QB:1,RB:2,WR:2,TE:1,FLEX:1,K:1,DST:1")
    ap.add_argument("--sims", type=int, default=1500, help="drafts simulated per slot/strategy")
    ap.add_argument("--adp-noise", type=float, default=0.20,
                    help="ADP std dev as a fraction of ADP (0.20 = realistic)")
    ap.add_argument("--tier-gap", type=float, default=15.0,
                    help="VOR gap that defines a tier break, in season points")
    ap.add_argument("--rb-weight", type=float, default=1.15,
                    help="RB score multiplier for the rb_priority strategy (1.0 = neutral)")
    ap.add_argument("--vona-weight", type=float, default=0.6,
                    help="weight on value-over-next-available vs raw VOR")
    ap.add_argument("--plan-rounds", type=int, default=10, help="rounds to detail in the plan")
    ap.add_argument("--plan-strategy", default="rb_priority", choices=STRATEGIES,
                    help="which strategy the round-by-round plan follows")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    if args.make_template:
        make_template(args.make_template)
        return 0
    if not args.players and not args.demo:
        ap.error("give --players PATH (or --make-template PATH, or --demo)")

    starters, flex = parse_starters(args.starters)
    lg = League(args.teams, args.rounds, starters, flex, args.adp_noise, args.tier_gap)

    cfg = Cfg()
    cfg.sims = args.sims
    cfg.rb_weight = args.rb_weight
    cfg.vona_weight = args.vona_weight
    cfg.candidates_per_pos = 4
    cfg.vona_depth = 25
    cfg.track_avail = True
    cfg.track_top = 24
    cfg.plan_rounds = args.plan_rounds

    if args.demo:
        players, skipped, derived, value_src = demo_pool(), [], 0, "synthetic demo pool"
    else:
        players, skipped, derived, value_src = load_players(args.players)

    base = compute_vor(players, lg)
    assign_tiers(players, lg)

    lines = []
    if args.demo:
        lines.append("!! DEMO MODE: synthetic players. Engine check only, not advice. !!")
        lines.append("")
    report_header(lg, players, base, skipped, derived, cfg, lines, value_src)
    report_cliffs(players, lg, lines)

    os.makedirs(args.outdir, exist_ok=True)
    write_board_csv(os.path.join(args.outdir, "board_vor.csv"), players)

    for slot in args.slots:
        picks = lg.pick_numbers(slot)
        lines.append("=" * 78)
        lines.append(f"SLOT {slot} -- pick schedule")
        lines.append("=" * 78)
        lines.append("  ".join(
            f"{lg.round_of(pk)}.{(pk - 1) % lg.teams + 1:02d}(#{pk})" for pk in picks))
        gap = picks[1] - picks[0]
        lines.append(f"Turn gap R1->R2: {gap} picks."
                     f"  R2->R3: {picks[2] - picks[1]} picks.")
        lines.append("")

        # Pass 1: availability curve, using the RB-priority reference strategy.
        base_res, _ = run_sims(players, lg, cfg, slot, "rb_priority",
                               surv=None, seed=args.seed + slot)
        surv = survival_table(base_res, picks)

        # Pass 2: VONA-aware strategy tournament on the same availability model.
        results = {}
        for k, strat in enumerate(STRATEGIES):
            res, _ = run_sims(players, lg, cfg, slot, strat, surv=surv,
                              seed=args.seed + 100 * (k + 1) + slot)
            results[strat] = res

        best = report_strategies(slot, results, lines)
        report_rb_check(slot, results, best, lines)
        # Plan follows the requested RB-priority strategy, not just the winner --
        # the tournament above quantifies what that preference costs.
        plan_strat = args.plan_strategy
        report_plan(slot, picks, results[plan_strat], surv, players, lg, cfg,
                    lines, plan_strat)

        write_availability_csv(os.path.join(args.outdir, f"availability_slot{slot}.csv"),
                               picks, surv, players, lg)

    lines.append("-" * 78)
    lines.append("FILES")
    lines.append("-" * 78)
    lines.append(f"{args.outdir}/board_vor.csv            full pool ranked by VOR, with tiers")
    for slot in args.slots:
        lines.append(f"{args.outdir}/availability_slot{slot}.csv   P(available) at each of your picks")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.outdir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
