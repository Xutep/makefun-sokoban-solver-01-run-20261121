# Sokoban solver — push-optimal A* search

Entry for the make.fun bounty `sokoban-solver-01-run-20261121` ("Build the best Sokoban solver, 21 November 2026").

`main.ts` (Node 22, stdlib only) solves each level room by room: the rooms of a generated level are independent (a box pushed out of a room can never return), so the level optimum is the sum of the rooms' optima. Each room is solved push-optimally with an A* over pushes — Hungarian box-to-goal assignment as the estimate, dead-square and frozen-2x2 pruning — and the rooms are stitched together with the player's walks between them. All 5 practice instances solve in their push-optimal counts (13, 16, 29, 47, 60 pushes).

Run: `node main.ts` reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt`.
