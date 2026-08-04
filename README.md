# 2026 Half-PPR Draft Model — slots 6 and 7

12-team, half-PPR, snake. RB-priority preference, quantified rather than assumed.

## Run it

```bash
python3 draft_model.py --players ringer_2026.csv --sims 3000 --slots 6 7
```

No dependencies beyond the Python standard library. Output goes to stdout and `out/`.

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `--slots 6 7` | `6 7` | Draft slots to model |
| `--sims 3000` | `1500` | Drafts simulated per slot per strategy |
| `--rb-weight 1.15` | `1.15` | RB thumb on the scale; `1.0` = neutral |
| `--plan-strategy` | `rb_priority` | Strategy the round-by-round plan follows |
| `--adp-noise 0.20` | `0.20` | How much your leaguemates deviate from ADP |
| `--starters` | `QB:1,RB:2,WR:2,TE:1,FLEX:1,K:1,DST:1` | Lineup shape |
| `--plan-rounds 8` | `10` | Rounds to detail |
| `--demo` | — | Synthetic pool; engine smoke test only |

## Data

`ringer_2026.csv` — all 224 players from The Ringer's 2026 preseason rankings
(Kelly / Heifetz / Horlbeck), updated Aug 3 2026, scraped from the half-PPR
draft tracker.

Columns: `name, pos, team, bye, adp, av, sos, pos_rank`.

**The Ringer publishes no projected points**, so the model derives value from
the auction value column (`av`). Auction dollars are themselves a
value-over-replacement calculation, so `AV → VOR` is close to a linear
transform — this preserves the rankers' positional scarcity judgement instead of
re-deriving it from ADP order. Consequence: absolute point totals in the report
are on a plausible-but-synthetic scale. **Rankings, VOR gaps, and tier breaks are
meaningful; the raw point totals are not.** Drop a `proj_pts` column into the CSV
and the model uses it instead, automatically.

176 of 224 players carry a nonzero AV. The $0 tail is ordered by ADP just below
replacement level.

## What the model does

1. **VOR with roster-aware baselines** — replacement level is the last projected
   league-wide starter at each position, with the single FLEX slot split
   45% RB / 50% WR / 5% TE.
2. **Tier detection** — breaks on a VOR gap whose threshold decays with
   positional depth, so late rounds don't collapse into one 50-player tier.
3. **Opponent simulation** — 11 opponents draft off ADP perturbed by noise
   proportional to ADP, with realistic roster limits (no third QB, no K/DST
   before the last two rounds).
4. **Availability curves** — P(player still on the board) at each of your picks.
5. **Strategy tournament** — `rb_priority`, `bpa`, `hero_rb`, `zero_rb`, each
   scored by the projected starting lineup it ends up with.
6. **VONA** — picks weigh a player against the expected best alternative
   surviving to your *next* pick, which is what actually drives snake-draft
   decisions.

## Output

| File | Contents |
|---|---|
| `out/report.txt` | The full text report |
| `out/board_vor.csv` | All 224 players ranked by VOR, with tiers |
| `out/availability_slot6.csv` | P(available) at each of your 16 picks |
| `out/availability_slot7.csv` | Same, slot 7 |

## Caveats

- Availability odds assume your league drafts near consensus ADP. A keeper
  league, a heavy homer, or an early QB run will shift them. Raise
  `--adp-noise` for a chaotic room.
- The model optimizes projected season points for the starting lineup. It does
  not model bye-week conflicts, injury risk, playoff schedule, or in-season
  trades. `bye` and `sos` are carried in the CSV for you to eyeball.
- Kicker and defense are treated as near-worthless until the final two rounds,
  which is correct for points but ignores streaming upside.
