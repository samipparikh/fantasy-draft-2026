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


## Post-draft report

The draft happened; `postdraft.py` grades it.

```bash
python3 postdraft.py --results hh_draft_2026.csv --focus KGB
```

It reads the draft-tracker export (`hh_draft_2026.csv`), scores every team by the
best starting lineup it can field in points over replacement, audits the focus
team pick by pick against the board it passed on, and enumerates every trade of
up to two players a side — keeping only the ones where *both* starting lineups
improve. Output is `docs/postdraft.json`, rendered by
[`docs/results.html`](https://samipparikh.github.io/fantasy-draft-2026/results.html).

Two things worth knowing about the scoring:

- **Defenses are excluded.** The export projects every DST between −69 and −190
  points, which is a different scale rather than a ranking.
- **An unfillable starting slot is charged the full replacement level**, so
  trading away your only quarterback never scores as a gain.

## Weekly defense model

`dst_model.py` projects every defense against the opponent it actually plays, for
all 18 weeks, and picks a start each week for one roster.

```bash
python3 dst_model.py --refresh                 # re-pull schedule + prior-season form
python3 dst_model.py --rostered TB,NYG,LV
```

Two components. **Points allowed**: an expected points-allowed line from an offense
index for the opponent, a defense index for the defense, and home field — the three
coefficients fit by least squares against the closing betting lines for the 202
team-games that have them, so the scale is the market's (mean absolute error 1.20
points). Expected tier points are then integrated over a normal centred on that
line, because the tier table is a steep staircase and points allowed vary around
the line with an SD of 9.0, measured from last season's results against the lines.
**Big plays**: sacks, takeaways and return scores as a linear function of the same
two indices around a 6.1-point league average.

Output is `docs/dst.json`, rendered by
[`docs/defense.html`](https://samipparikh.github.io/fantasy-draft-2026/defense.html).

Worth knowing:

- **All 32 defenses sit on one scale.** The draft export ranks only the defenses it
  contains (DEF1–DEF31); the 11 undrafted teams are exactly the 11 missing rank
  slots, so they are ordered by last season's points allowed, dropped into those
  slots, and given a projection interpolated off the rank curve. A prior built from
  last season alone rates Cleveland an above-average defense — the league's own
  board has it 17th.
- **Betting lines exist only through week 8.** Later weeks use the same fitted
  coefficients with no market to check them against, so they rank matchups rather
  than forecast points.

`data/nfl_2026_schedule.csv` and `data/nfl_2025_team_form.csv` are vendored slices
of [nflverse](https://github.com/nflverse/nfldata) `games.csv`, so a normal run is
offline; `--refresh` re-pulls them.

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
| `docs/postdraft.json` | Post-draft grades, pick audit and trade finder output |
| `docs/dst.json` | Weekly defense projections, start calls and model coefficients |
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
  which is correct for points but ignores streaming upside — `dst_model.py` is the
  answer to the defense half of that.
