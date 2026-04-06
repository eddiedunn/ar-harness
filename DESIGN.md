# ar-harness Skill System Design

## 1. Overview

The autoresearch protocol (Karpathy's pattern) has three moving parts: a system prompt (`program.md`) that defines the experiment loop, a mutable training script (`train.py`) the agent iterates on, and a fixed evaluation harness (`prepare.py`) that defines the metric contract. The `ar-harness` repo already provides the generic agent loop -- it reads `program.md`, grants the agent Read/Write/Edit/Bash tools, and runs until max turns or interruption.

What's missing is the layer above: generating experiment scaffolds from problem descriptions, and deploying them to GPU machines. Today this is done manually -- SSH in, clone repos, write program.md by hand, launch in tmux. The skill system automates this into two composable operations:

- **ar-design** -- generate a complete experiment directory from a problem description
- **ar-run** -- deploy an experiment directory to a remote GPU machine and manage execution

These are toolshed skills, usable from Claude Code on Mac, OpenClaw on trinity, or any future agent context. They orchestrate around `harness.py` -- they never replace it.

## 2. Skill Breakdown

### ar-design

**Purpose:** Transform a natural-language problem description into a ready-to-run experiment directory.

**Interface:**

| Input | Source | Required |
|-------|--------|----------|
| Problem description | `$ARGUMENTS` or conversation | Yes |
| Target metric | Inferred from problem, confirmed by user | Yes |
| Dataset source | Inferred or specified | Yes |
| Existing prepare.py | Path to user-provided prepare.py | No |
| Base model / architecture | Inferred or specified | No (defaults to transformer) |
| Time budget | Defaults to 300s | No |

| Output | Path |
|--------|------|
| `program.md` | `<experiment-dir>/program.md` |
| `train.py` | `<experiment-dir>/train.py` |
| `prepare.py` (candidate) | `<experiment-dir>/prepare.py` |
| `pyproject.toml` | `<experiment-dir>/pyproject.toml` |

**Operations:**

1. Parse the problem description. Identify: domain, dataset, metric, architecture constraints, hardware assumptions.
2. If the user provided an existing `prepare.py`, read it to extract: metric name, constants, exported interface. Skip steps 2-3 and proceed directly to step 4 using the extracted info.
3. Generate `prepare.py` candidate -- dataset loading, tokenizer/preprocessing, evaluation function, constants. Present to user.
4. **GATE: User confirms `prepare.py`.** Nothing else proceeds until this is approved. The user may edit it, request changes, or approve as-is. (Skipped when the user provided their own `prepare.py` -- they already own it.)
5. Generate `program.md` from the protocol template, injecting problem-specific sections.
6. Generate `train.py` scaffold with baseline architecture and training loop.
7. Generate `pyproject.toml` with dependencies.
8. Write all files to the experiment directory.

**Example invocation:**

```
/ar-design Pretrain a small GPT on FineWeb-Edu, optimize for val BPB on a held-out shard, single RTX 4090, 5 minute time budget
```

```
/ar-design Fine-tune a ViT-B/16 on CIFAR-100, optimize for top-1 val accuracy, single GPU, 10 minute budget
```

### ar-run

**Purpose:** Deploy an experiment directory to a remote GPU machine and manage the agent loop via tmux.

**Interface:**

| Input | Source | Required |
|-------|--------|----------|
| Experiment directory | `$ARGUMENTS` (local path) | Yes |
| SSH target | `$ARGUMENTS` or default (eddie@tela) | Yes |
| Remote base path | Default: `~/experiments` | No |
| Model | Default: `claude-sonnet-4-6` | No |
| Max turns | Default: 100 | No |

| Output | |
|--------|---|
| tmux session name | Printed on launch |
| Status updates | On-demand via `/ar-run status` |
| Results | Collected via rsync on completion |

**Operations:**

1. Validate remote prerequisites: `ssh <target> "which uv && which git && which tmux && git config user.name && git config user.email"`. If any check fails, print what's missing and abort with setup instructions. (The future `ar-setup` skill will automate this.)
2. Validate the local experiment directory has the required files (`program.md`, `train.py`, `prepare.py`, `pyproject.toml`).
3. rsync the experiment directory to `<ssh-target>:<remote-base>/<experiment-name>/`. Also rsync the ar-harness repo to `<ssh-target>:<remote-base>/ar-harness/` and rewrite the path dependency in the remote `pyproject.toml` to match.
4. SSH to the target. In a single SSH command chain:
   a. `cd` to the remote experiment directory.
   b. Create/sync the venv: `uv sync`.
   c. Run `uv run python prepare.py` to download datasets, train tokenizers, build caches. This can take minutes to hours depending on dataset size -- must complete before the agent loop starts.
   d. Launch in a named tmux session: `tmux new-session -d -s <experiment-name> 'CLAUDE_OAUTH_TOKEN=<token> uv run ar-harness --cwd . --max-turns <N> --model <M> 2>&1 | tee run.log'`
5. Print the tmux session name and connection command.

**Subcommands:**

| Subcommand | What it does |
|------------|-------------|
| `ar-run <dir> <target>` | Deploy and launch (default) |
| `ar-run status <target>` | List active tmux sessions, tail recent output |
| `ar-run tail <target> <session>` | `tmux capture-pane` + `tail run.log` |
| `ar-run stop <target> <session>` | `tmux send-keys C-c` then wait for graceful shutdown |
| `ar-run collect <target> <session>` | rsync results back to local machine |

**Example invocation:**

```
/ar-run ~/experiments/gpt-fineweb-apr04 eddie@tela
```

```
/ar-run status eddie@tela
```

```
/ar-run collect eddie@tela gpt-fineweb-apr04
```

## 3. ar-design: Experiment Generation

### Protocol decomposition

The existing `program.md` in the autoresearch repo mixes protocol (how the agent loop works) with problem specifics (GPT pretraining on climbmix). To make this generic, split into fixed protocol and variable problem sections:

**Fixed protocol (domain-agnostic):**

- Setup phase: branch creation, file reading, data verification, results.tsv init
- Experiment loop structure: modify train.py, commit, run, parse results, keep/discard
- Output format parsing (the `---` summary block)
- Logging rules: TSV format, columns, status values
- Git workflow: advance on improvement, reset on regression
- Autonomy rules: never stop, never ask, run indefinitely
- Simplicity criterion: complexity cost vs improvement magnitude
- Timeout and crash handling

**Variable problem sections:**

- What the agent is optimizing (metric name, direction -- lower/higher is better)
- What `train.py` contains and what's fair game to change
- What `prepare.py` provides (constants, data loading, evaluation function)
- Hardware context (GPU type, VRAM budget as soft constraint)
- Domain-specific hints (e.g., "consider learning rate schedules", "try different architectures")
- Time budget value
- Run command (usually `uv run train.py`)
- How to extract the metric from logs (grep pattern)

### program.md generation

The skill assembles `program.md` by:

1. Starting with the fixed protocol template (the loop rules, git workflow, autonomy mandate, logging format). This is invariant across all experiments.
2. Injecting a **Problem** section at the top that describes what's being optimized, in concrete terms the agent can act on.
3. Injecting a **Files** section that describes what the agent can and cannot modify.
4. Injecting **Evaluation** details: metric name, how it's computed, how to extract it from output.
5. Injecting **Hardware** context if provided.

The fixed protocol sections reference these variable sections by name (e.g., "get the lowest `{metric_name}`") so the template reads naturally.

### train.py scaffold generation

The scaffold provides a runnable baseline that the agent will iterate on. It includes:

**Boilerplate (all experiments):**
- Environment setup (`os.environ` flags, torch settings)
- Imports from `prepare.py` (constants, data loading, evaluation)
- Training loop skeleton: forward, backward, optimizer step, logging, time budget check
- Summary block printing at the end (metric, timing, VRAM, etc.)
- `torch.compile` wrapping
- GC management pattern (freeze after warmup)

**Problem-specific:**
- Model architecture (e.g., GPT, ViT, ResNet -- matched to the problem)
- Optimizer configuration (AdamW defaults, or Muon if transformer)
- Learning rate schedule
- Hyperparameter block at the top (clearly marked as "edit these")
- Data loading call matched to `prepare.py`'s interface

The scaffold must be a valid, runnable script that establishes a baseline on the first run. The agent's job is to improve from there.

For known domains (NLP pretraining, vision classification, fine-tuning), ar-design uses curated starter templates rather than generating from scratch. These templates are proven baselines with correct CUDA usage, proper distributed training patterns, and sane hyperparameter defaults. A real train.py for GPT pretraining is 600+ lines of sophisticated CUDA code -- an LLM cannot reliably produce that from a one-line problem description. Freeform generation is a fallback for novel domains where no template exists.

### prepare.py generation

This is the most sensitive file -- it defines the metric contract. It contains:

- **Constants:** `MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS` (or equivalents for non-NLP tasks)
- **Data loading:** Download/cache logic, dataset splits, dataloader factory
- **Tokenizer/preprocessing:** Whatever the task needs (tokenizer for NLP, transforms for vision, etc.)
- **Evaluation function:** The ground truth metric computation. Must be deterministic given a model and data.
- **Dependency spec:** What packages are needed (feeds into `pyproject.toml`)

The skill generates a candidate `prepare.py` and presents it to the user with:
- What metric it computes and how
- What dataset it uses and how it's split
- What constants are fixed
- What interface `train.py` imports from it

### The human gate

This is a hard requirement, not a suggestion. The gate exists because:

1. The metric defines what "better" means. A wrong metric means the agent optimizes the wrong thing for hours.
2. The dataset split defines what's train vs val. Leaking val into train makes results meaningless.
3. The evaluation function must be deterministic and correct. A bug here invalidates all results.
4. Dependencies must be pinned. A wrong torch version wastes GPU hours on install failures.

**Gate protocol:**
- Skill presents `prepare.py` with an explanation of the metric contract.
- User can: approve, request changes (skill regenerates), or edit manually.
- Only after explicit approval does the skill proceed to generate `program.md` and `train.py`.
- No marker file is needed -- if the user ran ar-design, the gate happened in conversation. ar-run just checks that `prepare.py` exists.

### Example: problem description to generated output

**Input:** "Pretrain a character-level language model on Shakespeare, optimize for val BPB, single GPU, 3 minute budget"

**Generated `prepare.py`:**
- Downloads Shakespeare text, splits 90/10 train/val by character position
- Character-level tokenizer (no BPE -- vocab is the character set)
- `MAX_SEQ_LEN = 256`, `TIME_BUDGET = 180`
- `evaluate_bpb()` computes bits per byte on val split
- Exports: `MAX_SEQ_LEN`, `TIME_BUDGET`, `Tokenizer`, `make_dataloader`, `evaluate_bpb`

**Generated `train.py`:**
- Small GPT (4 layers, 128 dim) as baseline
- AdamW optimizer, cosine schedule
- Training loop with time budget, summary block at end

**Generated `program.md`:**
- Fixed protocol (loop, git, logging, autonomy)
- Problem: "minimize val_bpb on Shakespeare character-level LM"
- Files: "modify only train.py, prepare.py is read-only"
- Evaluation: "metric is val_bpb (lower is better), extract with `grep '^val_bpb:' run.log`"

## 4. ar-run: Deployment and Execution

### SSH + tmux launch pattern

The proven pattern from manual runs:

```bash
# 1. Sync experiment to remote
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  <local-experiment-dir>/ <ssh-target>:<remote-base>/<name>/

# 2. Launch in tmux (single SSH command)
ssh <ssh-target> "
  cd <remote-base>/<name> &&
  uv sync --quiet &&
  uv run python prepare.py &&
  tmux new-session -d -s <name> \
    'CLAUDE_OAUTH_TOKEN=<token> uv run python -m ar_harness --cwd . --max-turns <N> --model <M> 2>&1 | tee run.log'
"
```

Key details:
- `--exclude='.venv'` -- never sync local venvs; let `uv sync` create the right one for the target's platform and CUDA version
- tmux session name matches experiment directory name (sanitized: alphanumeric + hyphens only)
- `tee run.log` captures everything while still letting tmux capture-pane work for the outer harness. Note: the GENERATED `program.md` must tell the inner agent to redirect stdout to file (`uv run train.py > run.log 2>&1`), not use tee -- per the protocol's explicit warning about flooding the agent's context with training output.
- The agent's stdout goes to both tmux scrollback and `run.log`

### OAuth token for remote execution

The claude-agent-sdk authenticates via `CLAUDE_OAUTH_TOKEN`, not `ANTHROPIC_API_KEY`. ar-run must ensure this token is available in the remote tmux session:

- For pre-configured targets (like tela), the token is already in the shell environment -- no injection needed.
- For rented boxes, ar-run injects it into the tmux command as an env var prefix: `CLAUDE_OAUTH_TOKEN=<token> uv run ...`
- The token is read from the local environment (`$CLAUDE_OAUTH_TOKEN`).
- **Security:** The token must never appear in `run.log` or `ps` output. Env var injection inside the tmux command string is safe -- tmux does not log its command args, and the env var is only visible in `/proc/<pid>/environ` (root-only). Never pass the token as a CLI argument.

### prepare.py execution

The `uv run python prepare.py` step runs before the agent loop starts. This step downloads datasets, trains tokenizers, builds caches, etc. It can take minutes to hours depending on dataset size. The step is conditional -- if the data cache already exists (e.g., from a previous run on the same machine), prepare.py should detect that and skip the download. ar-run runs it unconditionally and lets prepare.py decide whether work is needed.

### ar-harness availability on the target

ar-harness is not on PyPI. ar-run rsyncs the ar-harness repo to the remote target alongside the experiment directory, and the experiment's `pyproject.toml` uses a path dependency:

```toml
dependencies = [
    "ar-harness @ file:///home/user/ar-harness",
    # ... problem-specific deps
]
```

ar-run rewrites the path in `pyproject.toml` to match the remote layout before rsync (or immediately after, before `uv sync`). The harness is a thin wrapper (~130 lines) with two dependencies (`claude-agent-sdk`, `anyio`), so install is fast.

**Torch CUDA index:** The torch dependency needs a UV source override for the correct CUDA version. The real autoresearch repo uses `pytorch-cu128` as a named index. ar-design should generate the appropriate `[tool.uv.sources]` section based on the target GPU, or ar-run should patch it before rsync. Without this, `uv sync` pulls CPU-only torch and training runs 100x slower.

### Monitoring

**Status check:**
```bash
ssh <target> "tmux list-sessions 2>/dev/null; echo '---'; \
  for s in <remote-base>/*/run.log; do \
    echo \$s; tail -1 \$s; echo; \
  done"
```

**Tail output (live):**
```bash
ssh <target> "tmux capture-pane -t <session> -p -S -50"
```

This is better than `tail -f run.log` because it works through the skill's SSH-command model (no persistent connection needed).

**Metric extraction (quick check without full collect):**
```bash
ssh <target> "grep '^val_bpb:\|^status' <remote-base>/<name>/results.tsv 2>/dev/null; \
  tail -5 <remote-base>/<name>/results.tsv"
```

### Stop / cleanup

**Graceful stop:**
```bash
ssh <target> "tmux send-keys -t <session> C-c"
```

The harness handles SIGINT cleanly (prints "[harness] interrupted" and exits). The agent finishes its current tool call and stops. `run.log` and `results.tsv` are preserved.

**Force kill (if graceful doesn't work after 30s):**
```bash
ssh <target> "tmux kill-session -t <session>"
```

**Cleanup (remove experiment from remote):**
```bash
ssh <target> "rm -rf <remote-base>/<name>"
```

The skill should confirm before cleanup since results may not have been collected yet.

### Result collection

```bash
rsync -avz <ssh-target>:<remote-base>/<name>/results.tsv <local-experiment-dir>/
rsync -avz <ssh-target>:<remote-base>/<name>/run.log <local-experiment-dir>/
```

Optionally collect the full directory (includes the best `train.py` at HEAD of the git branch):
```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.cache' \
  <ssh-target>:<remote-base>/<name>/ <local-experiment-dir>/
```

The skill parses `results.tsv` after collection and presents a summary: best result, number of experiments run, keep/discard/crash counts.

### Multi-machine considerations

Nothing changes in the skill. Different SSH targets, same commands. Examples:

```
/ar-run ~/experiments/gpt-fineweb eddie@tela           # owned GPU
/ar-run ~/experiments/gpt-fineweb root@203.0.113.42    # Lambda instance
/ar-run ~/experiments/gpt-fineweb ubuntu@vast-abc123   # Vast.ai
```

For hourly rentals, the skill should note the launch time so the user can track cost. But the skill itself has no billing integration -- that's out of scope.

## 5. Experiment Directory Structure

A complete experiment directory on disk:

```
<experiment-name>/
  program.md          # System prompt for the agent (protocol + problem)
  train.py            # Mutable training script (agent modifies this)
  prepare.py          # Fixed evaluation harness (read-only during experiment)
  pyproject.toml      # Dependencies (includes ar-harness, torch, problem-specific)
  .python-version     # Python version pin (e.g., "3.11")
  results.tsv         # Experiment log (created by agent, not committed)
  run.log             # Stdout/stderr from current or last run
  .git/               # Git repo (agent commits to autoresearch/<tag> branch)
```

Files created during execution (by the agent, on the remote):

```
  .venv/              # Created by uv sync on the target machine
  .cache/             # Dataset cache (if prepare.py downloads data)
```

The `.git` repo is initialized by `ar-design` with an initial commit containing all generated files. The agent creates the `autoresearch/<tag>` branch as its first action per the protocol.

### Naming convention

Experiment directories are named: `<problem-short>-<date>[-<variant>]`

Examples:
- `gpt-fineweb-apr04`
- `vit-cifar100-apr04`
- `gpt-shakespeare-apr04-a100` (variant suffix for different hardware)

## 6. Future Considerations (not building now)

**ar-setup: machine provisioning checklist.** A non-automated checklist skill that verifies a remote machine is ready: CUDA toolkit, uv, claude CLI authenticated, tmux, sufficient disk space. Prints pass/fail for each item. No auto-install -- just diagnosis. Useful for rented instances where you SSH in cold.

**Multiple concurrent experiments.** Run different experiments on different machines simultaneously. The current design already supports this (each is an independent tmux session on a different target). What's missing is a dashboard: `/ar-run dashboard` that queries all known targets and shows active experiments, elapsed time, best metric so far. Needs a lightweight registry of active experiments (local JSON file).

**Automatic result comparison.** After collecting results from multiple runs (same problem, different machines or configs), generate a comparison table and plot. Parse `results.tsv` from each, align by experiment description, highlight the best overall result and the path that got there.

**Cost tracking for hourly rentals.** Record launch time, instance type, and hourly rate. On collect or stop, compute total cost. Integrate with the results summary so you can see cost-per-improvement. Needs per-provider rate tables or user-supplied rate.

**Experiment branching / forking.** Take the best `train.py` from a completed run and use it as the starting point for a new run with a different time budget, dataset, or hardware. The current git-branch model supports this naturally -- fork from the best commit.

**Warm-start on new hardware.** When moving an experiment from one machine to another (e.g., tela -> rented A100), carry over `results.tsv` and the current `train.py` state so the agent doesn't repeat already-tried ideas. This is just an rsync of the full directory -- the design already supports it, but the skill should recognize it's a continuation, not a fresh start.
