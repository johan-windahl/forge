# Walkthrough: the brief's own example

> Build a browser-based Quake-inspired FPS with one polished level.

This is a hard prompt for an autonomous system, and a good one to reason about,
because almost everything that matters is unstated. What follows is what Forge
does with it and — more usefully — *why*, with the places it is likely to struggle
called out honestly.

---

## Setup

```bash
mkdir fps && cd fps
forge init "Build a browser-based Quake-inspired FPS with one polished level."
```

Recommended configuration for a project like this:

```toml
# .forge/config.toml
[budget]
total_cost = 150.0
daily_cost = 40.0
cloud_fraction_target = 0.12

[scheduler]
workers = 2
lease_seconds = 2400          # 3D asset generation and builds are slow

[sandbox]
kind = "docker"               # this will run a lot of generated build scripts

[validation]
gates = ["schema", "secrets", "lint", "types", "build", "unit"]
visual_tolerance = 0.02       # a rendered 3D scene is not pixel-stable

[validation.gate_settings.load_perf]
budget_ms = 3000              # a game that takes 8s to load is not polished

[deploy]
enabled = true
strategy = "static"
```

Two of those deserve a note.

**`visual_tolerance = 0.02`** — the default 0.5% is right for a UI and wrong for a
3D scene, where lighting and anti-aliasing produce small differences between
otherwise-identical renders. Too tight and every screenshot triggers a review;
too loose and a genuine regression slides through.

**`load_perf.budget_ms`** — "polished" is subjective, but load time is not. Making
one component of polish measurable is worth more than any amount of prompting
about it.

---

## What the planner decides for you

Run it and read `forge memory --kind assumption`. On a prompt this open, expect
roughly:

| Assumption | Why it has to be made |
| --- | --- |
| Three.js over raw WebGL | "Browser-based 3D" with no stated constraint; the ecosystem answer |
| Single-player, no networking | "Quake-inspired" is about feel; multiplayer was not requested |
| Keyboard + mouse with pointer lock | The genre's convention |
| One level means one connected arena | Rather than a level *editor*, which would be a different project |
| Placeholder geometry, no external art | It cannot download assets; it must be honest about that |

Every one of those is a decision a human would otherwise be asked about. They are
recorded with a confidence and a revisit condition, which is the point: you can
read them, disagree with one, and correct it:

```bash
forge memory --kind assumption
forge unblock <node> "Use raw WebGL, no framework dependency."
```

The last row is the one to watch. "Polished" with programmer-art geometry is a
real tension, and it is exactly where the goal check will push back.

---

## Milestones it typically produces

1. **Playable shell** — scaffold, render loop, first-person camera, pointer lock.
   Ends with something that runs in a browser and responds to input.
2. **Movement and collision** — the part that determines whether it feels like
   Quake. Air control, ground friction, step-up geometry.
3. **Combat** — weapon, projectiles or hitscan, targets, damage.
4. **The level** — arena geometry, spawn points, pickups, lighting.
5. **Polish** — HUD, audio, performance budget, visual pass.

The first milestone ending in something *runnable* is not incidental. A milestone
that ends with untested scaffolding gives the validation layer nothing to check,
so problems surface much later and much more expensively. The planner is
instructed on exactly this.

---

## Where the interesting verification happens

A game is the case where "the tests pass" says almost nothing.

**Browser QA** designs a flow from acceptance criteria and runs it in real
Chromium. For the shell milestone that is: load the page, confirm a canvas
exists, confirm no console errors, confirm the pointer-lock prompt appears. The
blank-page check matters more here than anywhere — a WebGL app that fails to
initialise renders nothing, throws nothing, and passes every unit test.

**Visual review** looks at captured frames and judges whether they show a game or
a debug scene. This is where placeholder geometry gets flagged, which is the
correct outcome: the tension between "polished" and "no external art" should
surface as a recorded finding rather than being silently resolved.

**Performance** measures load time and frame timing against the configured
budget. A 3D scene that loads in nine seconds fails a check rather than an
opinion.

---

## Expected cost and where it goes

For a project of this size, after the first milestone the steady state is roughly:

- **Planning and architecture** — small volume, high per-call cost. These route to
  frontier models deliberately, and it is the right trade: a bad module boundary
  costs a hundred times more than the tokens saved planning cheaply.
- **Implementation** — the bulk of the volume, mostly local once the router has
  evidence.
- **Debugging** — the class most likely to escalate. Physics and collision bugs
  are where local models struggle most, and the router discovers that on its own.
  `forge policy` will show it.
- **Visual review** — small volume, and the only vision-model spend. It runs only
  when the pixel comparison reports a change.

```bash
forge metrics    # by_task_class shows exactly where it went
```

If the first milestone looks cloud-heavy, that is expected — the routing priors
have not been corrected by evidence yet. Check again after milestone two.

---

## Where it will struggle

Stated plainly, because knowing this in advance is worth more than optimism.

**"Polished" is the hard word.** Forge can verify a game loads, runs at a target
frame rate, has no console errors and matches a visual baseline. It cannot verify
that the movement *feels* right, which is most of what makes a Quake-inspired
shooter good. Expect the goal check to accept something a player would call
functional but unremarkable.

**Game feel is tuning, not code.** Air acceleration, friction, jump height — these
are numbers found by playing. There is no gate for them. The most effective
intervention is to give the numbers as a requirement:

```bash
forge unblock <node> "Movement targets: ground accel 10, air accel 1, friction 6, jump 270ups."
```

**Assets.** It will produce placeholder geometry and say so. If you want real
art, that is a human contribution, and the honest read of the assumption record is
that Forge already told you.

**Long builds interact with lease timing.** 3D asset pipelines are slow. If you
see nodes being reclaimed mid-work, raise `lease_seconds`.

---

## Reading the result

```bash
forge status --verbose
forge report --open              # dashboard with embedded screenshots
forge memory --export MEMORY.md  # every assumption and decision, in one file
cd workspace && git log --oneline
```

Each commit carries a `Forge-Node` trailer, so the history is traceable back to
the task that produced it — which is what makes a three-day autonomous run
reviewable by a person afterwards.
