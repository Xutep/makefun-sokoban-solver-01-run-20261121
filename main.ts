// Sokoban solver: push-optimal search, room by room.
//
// Every level this competition draws is a hallway with one to four small
// rooms off it, each room holding 3 or 4 boxes and as many goals, and a box
// pushed out of a room can never come back -- so the rooms are independent
// problems and the level's push optimum is the sum of the rooms' optima.
// This program finds each room, solves it push-optimally with an A* over
// pushes (Hungarian box-to-goal assignment as the estimate, dead-square and
// frozen 2x2 pruning), and stitches the rooms together with the player's
// walks between them. Everything is budgeted by work counted, never by the
// clock, so two runs write byte-identical answers.
//
// Node 22, standard library only, no network, no randomness.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (state counts, never the clock): a room the generator keeps
// is solvable well inside these, and the cap keeps the whole 5-level set
// inside the 60-second wall even on a slow machine.
const MAX_STATES_PER_SEARCH = 1_000_000;
const MAX_STATES_PER_RUN = 4_000_000;
let statesUsed = 0;

const DIRS = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
] as const;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The squares a box can ever stand on and still reach a goal: found by
 * pulling boxes back from the goals over open floor. A box off a live
 * square is dead, so no search state containing one can ever be solved.
 */
function liveSquares(lv: Level): Uint8Array {
  const { w, wall, goals } = lv;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const d of DIRS) {
      const step = d.dy * w + d.dx;
      const y = x - step; // the square a box is pulled back onto
      if (!wall[y] && !wall[y - step] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Flood fill from `from` around walls and occupied squares; returns the lowest square reached. */
function floodLow(
  lv: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = lv;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(lv: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = lv;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** Minimum total Manhattan distance of an assignment of boxes to goals (bitmask DP; boxes.length <= 4 here). */
function hungarian(boxes: number[], goals: number[], w: number): number {
  const n = boxes.length;
  const cost: number[][] = boxes.map((b) => goals.map((g) => Math.abs((b % w) - (g % w)) + Math.abs(Math.floor(b / w) - Math.floor(g / w))));
  const full = (1 << n) - 1;
  const dp = new Array<number>(1 << n).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask <= full; mask++) {
    const k = popcount(mask); // boxes[0..k) already assigned
    if (k >= n) continue;
    for (let j = 0; j < n; j++) {
      if (mask & (1 << j)) continue;
      const v = dp[mask]! + cost[k]![j]!;
      if (v < dp[mask | (1 << j)]!) dp[mask | (1 << j)] = v;
    }
  }
  return dp[full]!;
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

interface SearchResult {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal A* over one level (a room, or a whole level as a fallback).
 * A state is the sorted boxes plus the player's reachable region, named by
 * its lowest square; the cost is pushes made, the estimate the Hungarian
 * assignment cost, which one push can lower by at most 1, so it is
 * admissible and the first solved state popped is push-optimal.
 * Returns null when the work budget runs out or the level has no solution.
 */
function solvePushes(lv: Level, maxStates: number): SearchResult | null {
  const { w, wall, goals } = lv;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(lv);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (lv.boxes.some((b) => !live[b])) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size); // this state's player region
  const mark = new Int32Array(size); // a successor's player region
  const stack = new Int32Array(size);
  let stamp = 0;

  const stateBoxes: number[][] = [lv.boxes.slice()];
  const statePlayer: number[] = [lv.player];
  const stateG: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1]; // the box square before the push that made this state
  const pushDir: number[] = [-1];
  const best = new Map<string, number>();
  for (const b of lv.boxes) occupied[b] = 1;
  const startLow = floodLow(lv, occupied, lv.player, seen, ++stamp, stack);
  for (const b of lv.boxes) occupied[b] = 0;
  const keyOf = (boxes: number[], low: number): string => boxes.join(',') + '@' + low;
  best.set(keyOf(lv.boxes, startLow), 0);

  // Binary heap of [f, -g, state]: lowest f first, deeper (higher g) first on ties.
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  heapPush([hungarian(lv.boxes, goals, w), 0, 0]);
  let found = solved(lv.boxes) ? 0 : -1;

  while (heap.length > 0 && found < 0) {
    const [, negG, s] = heapPop();
    const g = -negG;
    const boxes = stateBoxes[s]!;
    for (const b of boxes) occupied[b] = 1;
    const low = floodLow(lv, occupied, statePlayer[s]!, seen, ++stamp, stack);
    if (best.get(keyOf(boxes, low)) !== s) {
      for (const b of boxes) occupied[b] = 0;
      continue;
    }
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const d = DIRS[di]!;
        const step = d.dy * w + d.dx;
        const t = b + step;
        if (seen[b - step] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(lv, occupied, goal, t);
        let next: number[] | null = null;
        let low = -1;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          low = floodLow(lv, occupied, b, mark, ++stamp, stack);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const k = keyOf(next, low);
        const prev = best.get(k);
        if (prev !== undefined && stateG[prev]! <= g + 1) continue;
        if (stateBoxes.length >= maxStates || statesUsed >= MAX_STATES_PER_RUN) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        statesUsed++;
        best.set(k, stateBoxes.length);
        stateBoxes.push(next);
        statePlayer.push(b);
        stateG.push(g + 1);
        parent.push(s);
        pushFrom.push(b);
        pushDir.push(di);
        if (solved(next)) {
          found = stateBoxes.length - 1;
          break;
        }
        heapPush([g + 1 + hungarian(next, goals, w), -(g + 1), stateBoxes.length - 1]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // Rebuild the push path, then the moves: walk to each push square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes.slice();
  for (const s of path) {
    const nextBoxes = stateBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const di = pushDir[s]!;
    const d = DIRS[di]!;
    const step = d.dy * w + d.dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(lv, occupied, player, from - step);
    for (const b of boxes) occupied[b] = 0;
    moves += d.push;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(lv: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = lv;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when some part holds more or fewer boxes than
 * goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(lv: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const live = liveSquares(lv);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const d of DIRS) {
        const q = squares[i]! + d.dy * w + d.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Solve one level's text into its move string; '' when the search gives up (still a valid, scored-0 answer). */
function solveLevel(text: string): string {
  const lv = parseLevel(text);
  if (lv.boxes.every((b) => lv.goals.includes(b))) return '';
  const rooms = roomsOf(lv);
  if (rooms !== null) {
    let player = lv.player;
    let moves = '';
    for (const room of rooms) {
      const wall = lv.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: lv.w,
        h: lv.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solved = solvePushes(sub, MAX_STATES_PER_SEARCH);
      if (solved === null) return '';
      moves += solved.moves;
      player = endOf(lv.w, player, solved.moves);
      room.done = true;
    }
    return moves;
  }
  const whole = solvePushes(lv, MAX_STATES_PER_SEARCH);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
