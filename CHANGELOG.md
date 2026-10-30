## v0.1.3 (2026-10-29)

### Fix

- **block_balance**: sequential stages and success only for a free-standing balanced beam
- **join_blocks**: require the peg to be inside the socket for success
- **sim**: enlarge the MuJoCo arena to avoid mj_stackAlloc aborts
- **assets**: stop re-downloading assets on every import
- **join_blocks**: disable midpoint integration to stop wall divergence on reset

### Refactor

- **assets**: rely on rcs-core 0.7.4 for asset versioning

## v0.1.2 (2026-10-08)

### Feat

- **pour_marbles**: pouring direction in instruction
- configure in which cup the marbles spawn

### Fix

- **pour_marbles**: pass task config to wrapper
- **ball_maze**: typo in task instructions for ball maze
- ball maze instruction typo

## v0.1.1 (2026-10-06)

### Fix

- monotone milestones

## v0.1.0 (2026-06-27)

### Fix

- assets path
