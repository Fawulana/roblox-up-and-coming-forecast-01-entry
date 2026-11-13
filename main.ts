// Roblox Up-and-Coming top-10 forecaster: Monte Carlo simulation of rank dynamics.
//
// The chart is a fast-moving rank order, so this entry forecasts it the way the
// chart itself moves: every game carries a latent position x = ln(rank), and the
// program simulates that position hour by hour over the scored window, many
// times, with a fixed seed, then counts how often each game lands on each of
// the top 10 places. The simulation's ingredients are all estimated from the
// history it is given:
//
//   - Diffusion: a game's log-rank wanders with a per-hour volatility that is
//     larger near the top of the chart (a move from #1 to #3 is a big log move)
//     and smaller deep in it, scaled up from the history's consecutive-read
//     moves after backtesting showed the raw values overconfident over 72h.
//   - Momentum: each game's recent log-rank velocity, shrunk hard (consecutive
//     moves correlate only ~0.2) and faded over about a day, so a climber keeps
//     some of its climb without the forecast chasing noise.
//   - Mean reversion: a gentle pull toward the game's own recency-weighted mean
//     rank, which keeps the 72-hour spread honest instead of letting diffusion
//     flatten every game into every place.
//   - Exits: games leave the chart, rarely from the top and often from the
//     bottom; the exit hazard grows with rank. Games whose recorded creation
//     date puts them near the chart's 28-day age limit get an extra hazard.
//   - Entries: new games join at roughly the history's rate, at ranks drawn
//     from the history's entry-rank distribution. They are never named, so the
//     places they take in the simulation stay in each place's unnamed share,
//     which is exactly the share that covers real new entrants at scoring time.
//
// Player counts are deliberately unused: in the history, a game's player-count
// growth has ~zero correlation with its next rank move.
//
// The run is fully deterministic: one fixed-seed generator drives every draw,
// ties break on universe id, and the output rows are written in a fixed order,
// so two runs on the same history write the same file byte for byte. Standard
// library only, no network, no model calls; it reads $MAKEFUN_INPUT_DIR (else
// /input) and writes $MAKEFUN_OUTPUT_DIR/forecast.json (else /output).
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- Model parameters, estimated from the chart's history. ---
const SIMS = 600; // Monte Carlo repetitions of the whole window.
const SEED = 0x51d15ea5; // fixed seed: the run must be byte-identical every time.
const SIGMA_TOP = 0.224; // log-rank volatility per sqrt(hour) at the top of the chart.
const SIGMA_BASE = 0.112; // ... deep in the chart.
const SIGMA_DECAY = 15; // rank scale over which volatility falls from top to base.
// The raw consecutive-read estimates are 0.14 top / 0.07 deep, but backtesting
// over 72-hour windows showed those values overconfident: the chart's moves
// compound faster than a per-hour random walk implies, so both are scaled 1.6x.
const KAPPA = 0.003; // mean reversion per hour toward the game's own mean log-rank.
const ANCHOR_HALF_LIFE_H = 48; // recency half-life for the mean-rank anchor.
const DRIFT_HALF_LIFE_H = 48; // recency half-life for the momentum estimate.
const DRIFT_SHRINK = 0.4; // shrinkage on estimated momentum (it is mostly noise).
const DRIFT_MAX = 0.02; // clip on momentum, in log-rank per hour.
const DRIFT_TAU_H = 24; // momentum decay time: drift fades as exp(-hour / tau).
const ENTRY_RATE = 0.78; // new games joining the chart per hour.
const ENTRY_LOG_RANK = 4.47; // median entry rank is ~87, ln(87) = 4.47.
const ENTRY_LOG_SD = 0.55; // spread of entry log-ranks.
const ENTRY_MIN_RANK = 15; // new games essentially never debut in the top 10.
const ENTRY_MAX_RANK = 140;
const EXIT_BASE = 0.001; // hourly exit hazard at the top of the chart.
const EXIT_TOP_UP = 0.017; // extra hazard deep in the chart.
const EXIT_MID_RANK = 60; // rank at which the hazard is halfway up.
const EXIT_RANK_SCALE = 8; // steepness of the hazard's rise with rank.
const AGE_HAZARD_START_D = 27; // age in days where the 28-day limit starts to bite.
const AGE_HAZARD_RATE = 0.25; // extra hourly hazard per day past the start.
const AGE_HAZARD_MAX = 0.5;
const EXITED_RANK = 101; // rank charged to an exited game when ordering the top 10.
const MIN_P = 0.0015; // rows below this are left to the place's unnamed share.
const MAX_ROWS_PER_PLACE = 100;
// Unnamed reserve: the share of each place left for games not on the chart when
// the history ends. Estimated from the history itself: over read pairs 12-72h
// apart, the pooled fraction of top-10 slots held by games absent from the
// earlier read's chart. New games almost never take #1 but often take the
// lower places, so the reserve rises gently with place number.
const RESERVE_MIN_GAP_H = 12;
const RESERVE_MAX_GAP_H = 72;
const RESERVE_DEFAULT = 0.1;
const RESERVE_CLIP_LO = 0.03;
const RESERVE_CLIP_HI = 0.25;

interface ChartRow {
  rank: number;
  universe_id: number;
  playing?: number | null;
}

interface Snapshot {
  at: string;
  up_and_coming: ChartRow[];
  details?: Record<string, { created?: string | null } | null> | null;
}

interface Task {
  places: number;
  output: string;
  history_until: string;
  window: { slots: number; slot_minutes: number };
}

// Deterministic PRNG (mulberry32) with a Box-Muller gaussian on top.
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Gaussian {
  private spare: number | null = null;
  private readonly uniform: () => number;
  constructor(uniform: () => number) {
    this.uniform = uniform;
  }
  next(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    do {
      u = this.uniform();
    } while (u <= 0);
    v = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    const t = 2 * Math.PI * v;
    this.spare = r * Math.sin(t);
    return r * Math.cos(t);
  }
}

function poisson(rate: number, uniform: () => number): number {
  const limit = Math.exp(-rate);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= uniform();
  } while (p > limit);
  return k - 1;
}

function sigmaForRank(rank: number): number {
  return SIGMA_BASE + (SIGMA_TOP - SIGMA_BASE) * Math.exp(-(rank - 1) / SIGMA_DECAY);
}

function exitHazard(rank: number, ageDays: number | null, hour: number): number {
  let h = EXIT_BASE + EXIT_TOP_UP / (1 + Math.exp(-(rank - EXIT_MID_RANK) / EXIT_RANK_SCALE));
  if (ageDays !== null) {
    const age = ageDays + hour / 24;
    if (age > AGE_HAZARD_START_D) {
      h += Math.min(AGE_HAZARD_MAX, (age - AGE_HAZARD_START_D) * AGE_HAZARD_RATE);
    }
  }
  return Math.min(0.9, h);
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const places = task.places ?? 10;
const hours = task.window?.slots ?? 72;
const historyUntil = Date.parse(task.history_until);

const snapshots = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((line) => line.trim() !== '')
  .map((line) => JSON.parse(line) as Snapshot)
  .sort((a, b) => Date.parse(a.at) - Date.parse(b.at));

const last = snapshots[snapshots.length - 1]!;
const lastTime = Date.parse(last.at);

// Pooled new-entrant share of top-10 slots, from read pairs 12-72h apart.
let reservePooled = RESERVE_DEFAULT;
{
  const chartSets = snapshots.map((s) => new Set(s.up_and_coming.map((g) => g.universe_id)));
  const tops = snapshots.map((s) => [...s.up_and_coming].sort((a, b) => a.rank - b.rank).slice(0, places));
  let slots = 0;
  let fresh = 0;
  for (let i = 0; i < snapshots.length; i++) {
    for (let j = i + 1; j < snapshots.length; j++) {
      const gapH = (Date.parse(snapshots[j]!.at) - Date.parse(snapshots[i]!.at)) / 3600000;
      if (gapH < RESERVE_MIN_GAP_H || gapH > RESERVE_MAX_GAP_H) continue;
      for (const g of tops[j]!) {
        slots++;
        if (!chartSets[i]!.has(g.universe_id)) fresh++;
      }
    }
  }
  if (slots > 0) reservePooled = fresh / slots;
}
const reserveForPlace = (place: number): number =>
  Math.max(RESERVE_CLIP_LO, Math.min(RESERVE_CLIP_HI, reservePooled * (0.5 + 0.1 * place)));

// Per-game history: observations of (time in hours, log-rank), plus age.
interface GameHistory {
  obs: Array<{ t: number; x: number }>;
  createdMs: number | null;
}
const histories = new Map<number, GameHistory>();
for (const snap of snapshots) {
  const t = Date.parse(snap.at) / 3600000;
  const rows = [...snap.up_and_coming].sort((a, b) => a.rank - b.rank);
  for (const row of rows) {
    let h = histories.get(row.universe_id);
    if (!h) {
      h = { obs: [], createdMs: null };
      histories.set(row.universe_id, h);
    }
    h.obs.push({ t, x: Math.log(row.rank) });
  }
  if (snap.details) {
    for (const [idText, detail] of Object.entries(snap.details)) {
      const id = Number(idText);
      const created = detail?.created;
      if (created) {
        const h = histories.get(id);
        if (h) h.createdMs = Date.parse(created);
      }
    }
  }
}

interface SimGame {
  id: number;
  x: number;
  anchor: number;
  drift: number;
  ageDays: number | null;
  alive: boolean;
  named: boolean;
}

const games: SimGame[] = [];
{
  const lastRows = [...last.up_and_coming].sort((a, b) => a.rank - b.rank);
  const tLast = lastTime / 3600000;
  for (const row of lastRows) {
    const h = histories.get(row.universe_id)!;
    const obs = h.obs;
    // Recency-weighted mean log-rank (the anchor).
    let wSum = 0;
    let wxSum = 0;
    for (const o of obs) {
      const ageH = Math.max(0, tLast - o.t);
      const w = Math.exp((-ageH * Math.LN2) / ANCHOR_HALF_LIFE_H);
      wSum += w;
      wxSum += w * o.x;
    }
    const anchor = wxSum / wSum;
    // Weighted slope of log-rank on time (momentum), centred, then shrunk.
    let drift = 0;
    if (obs.length >= 3) {
      let wtSum = 0;
      let dtBar = 0;
      let xBar = 0;
      for (const o of obs) {
        const ageH = Math.max(0, tLast - o.t);
        const wd = Math.exp((-ageH * Math.LN2) / DRIFT_HALF_LIFE_H);
        wtSum += wd;
        dtBar += wd * (o.t - tLast);
        xBar += wd * o.x;
      }
      dtBar /= wtSum;
      xBar /= wtSum;
      let num = 0;
      let den = 0;
      for (const o of obs) {
        const ageH = Math.max(0, tLast - o.t);
        const wd = Math.exp((-ageH * Math.LN2) / DRIFT_HALF_LIFE_H);
        const dt = o.t - tLast - dtBar;
        num += wd * dt * (o.x - xBar);
        den += wd * dt * dt;
      }
      if (den > 0) drift = (num / den) * DRIFT_SHRINK;
    }
    drift = Math.max(-DRIFT_MAX, Math.min(DRIFT_MAX, drift));
    const ageDays =
      h.createdMs !== null && Number.isFinite(historyUntil)
        ? (historyUntil - h.createdMs) / 86400000
        : null;
    games.push({
      id: row.universe_id,
      x: Math.log(row.rank),
      anchor,
      drift,
      ageDays,
      alive: true,
      named: true,
    });
  }
}

// --- Monte Carlo: simulate the window, count place occupancy. ---
const occupancy = new Map<number, number[]>();
for (const g of games) occupancy.set(g.id, new Array<number>(places).fill(0));
const rankSum = new Map<number, number>();
for (const g of games) rankSum.set(g.id, 0);
const unnamedOccupancy = new Array<number>(places).fill(0);

for (let sim = 0; sim < SIMS; sim++) {
  const uniform = makeRng((SEED + sim * 0x9e3779b9) >>> 0);
  const gauss = new Gaussian(uniform);
  // Fresh copy of the starting state for this repetition.
  const state: SimGame[] = games.map((g) => ({ ...g }));
  let nextNewId = -1;
  for (let hour = 0; hour < hours; hour++) {
    const driftFade = Math.exp(-hour / DRIFT_TAU_H);
    // New entrants join before the hour's ranking.
    const nNew = poisson(ENTRY_RATE, uniform);
    for (let k = 0; k < nNew; k++) {
      let x0 = ENTRY_LOG_RANK + ENTRY_LOG_SD * gauss.next();
      x0 = Math.max(Math.log(ENTRY_MIN_RANK), Math.min(Math.log(ENTRY_MAX_RANK), x0));
      state.push({ id: nextNewId--, x: x0, anchor: x0, drift: 0, ageDays: 0, alive: true, named: false });
    }
    for (const g of state) {
      if (!g.alive) continue;
      const rank = Math.exp(g.x);
      g.x += g.drift * driftFade - KAPPA * (g.x - g.anchor) + sigmaForRank(rank) * gauss.next();
      if (uniform() < exitHazard(rank, g.ageDays, hour)) g.alive = false;
    }
    const alive = state.filter((g) => g.alive);
    alive.sort((a, b) => (a.x - b.x === 0 ? a.id - b.id : a.x - b.x));
    const posOf = new Map<SimGame, number>();
    for (let k = 0; k < alive.length; k++) posOf.set(alive[k]!, k + 1);
    for (let k = 0; k < Math.min(places, alive.length); k++) {
      const g = alive[k]!;
      if (g.named) occupancy.get(g.id)![k]++;
      else unnamedOccupancy[k]++;
    }
    for (const g of state) {
      if (!g.named) continue;
      rankSum.set(g.id, rankSum.get(g.id)! + (posOf.get(g) ?? EXITED_RANK));
    }
  }
}

// --- Build the forecast. ---
const total = SIMS * hours;
interface ProbRow {
  universe_id: number;
  place: number;
  p: number;
}
const probabilities: ProbRow[] = [];
for (let place = 1; place <= places; place++) {
  // Scale the simulated shares so the place keeps its unnamed reserve.
  const simSum = [...occupancy.values()].reduce((s, counts) => s + counts[place - 1]! / total, 0);
  const scale = simSum > 0 ? (1 - reserveForPlace(place)) / simSum : 0;
  const rows: ProbRow[] = [];
  for (const [id, counts] of occupancy) {
    const p = (counts[place - 1]! / total) * scale;
    if (p >= MIN_P) rows.push({ universe_id: id, place, p: Math.round(p * 1e6) / 1e6 });
  }
  rows.sort((a, b) => b.p - a.p || a.universe_id - b.universe_id);
  const kept = rows.slice(0, MAX_ROWS_PER_PLACE);
  // Guard the per-place sum: rounding must never push it over 1.
  const sum = kept.reduce((s, r) => s + r.p, 0);
  if (sum > 1 && kept.length > 0) {
    kept[0]!.p = Math.round((kept[0]!.p - (sum - 1)) * 1e6) / 1e6;
  }
  probabilities.push(...kept);
}
probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);

// Top 10 by expected rank over the simulated window (exited hours count 101).
const expected = [...rankSum.entries()]
  .map(([id, s]) => ({ id, mean: s / total }))
  .sort((a, b) => a.mean - b.mean || a.id - b.id);
const top10 = expected.slice(0, places).map((e) => e.id);

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output ?? 'forecast.json'), `${JSON.stringify({ probabilities, top10 })}\n`);
