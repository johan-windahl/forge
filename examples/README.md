# Examples

| Example | What it shows |
| --- | --- |
| [`fps-game/`](fps-game/) | The brief's own example, walked through end to end |
| [`embedding-forge.py`](embedding-forge.py) | Driving Forge from Python instead of the CLI |
| [`custom-gate.py`](custom-gate.py) | Adding a project-specific validation gate |
| [`configs/`](configs/) | Configurations for common situations |

---

## The thirty-second version

```bash
mkdir clock && cd clock
forge init "A single-page web app showing an analog clock with a dark mode toggle."
forge run --dry-run    # exercise everything, spend nothing
forge run              # build it
forge status
forge report --open
```

## What actually happens

1. **`init`** creates the git workspace, records the goal as a requirement,
   and adds two nodes: a `plan` node and a `goal` barrier node.

2. **The planner** interprets the goal, writes down its assumptions (which
   framework, whether persistence is needed, what "dark mode toggle" implies),
   defines milestones, and emits the task graph for the first one — including
   its own validation, review and retrospective nodes.

3. **The architect** decides the stack and module boundaries, and writes
   `docs/architecture.md` into the project along with interface records that let
   later nodes implement against boundaries without reading each other's code.

4. **Implementation nodes** run — concurrently where they do not depend on each
   other. Each produces an edit plan, applies it atomically, runs the gates, and
   fixes what it can before handing failure back to the scheduler.

5. **Browser QA** designs an interaction flow from the acceptance criteria, runs
   it in real Chromium, captures screenshots, and files what broke.

6. **Visual review** looks at the screenshots — but only if the pixel comparison
   says something changed.

7. **The goal barrier** waits until nothing else can progress, then re-reads the
   *original sentence* and judges whether it was delivered. If not, it creates
   gap-closing work and another barrier.

8. **The retrospective** computes what the milestone actually cost, where the
   time went, and what should change — then records lessons that outlive the
   project.

## Watching it

```bash
forge watch                 # in a second terminal
forge status --verbose      # progress, spend, routing
forge memory --kind assumption   # what it decided on your behalf
forge metrics               # where the money and the time went
```
