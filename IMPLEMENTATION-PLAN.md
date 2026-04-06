# ar-design + ar-run Skills: Implementation Plan

## What you're building

Two Claude Code toolshed skills that automate the [Karpathy autoresearch protocol](https://github.com/karpathy/autoresearch):

- **ar-design** — generate a complete experiment directory (program.md, train.py, prepare.py, pyproject.toml) from a natural-language problem description
- **ar-run** — deploy that directory to a remote GPU machine via SSH + tmux and manage execution

Both skills live in `/Users/tmwsiy/code/toolshed/skills/` — the central toolshed repo, not claw-deploy.

## Repos involved

- `/Users/tmwsiy/code/ar-harness/` — thin Claude Agent SDK harness (~130 lines, already written)
- `/Users/tmwsiy/code/autoresearch/` — Karpathy's upstream repo (reference for protocol + file formats)
- `/Users/tmwsiy/code/toolshed/skills/` — where skills live; read 2-3 existing skills to learn the format

## Files to create / modify

| File | Action |
|------|--------|
| `/Users/tmwsiy/code/ar-harness/pyproject.toml` | Add `[project.scripts]` entry point |
| `/Users/tmwsiy/code/toolshed/skills/ar-design/SKILL.md` | Create |
| `/Users/tmwsiy/code/toolshed/skills/ar-design/references/program-protocol.md` | Create |
| `/Users/tmwsiy/code/toolshed/skills/ar-run/SKILL.md` | Create |

---

## Step 0: Fix ar-harness entry point

`/Users/tmwsiy/code/ar-harness/pyproject.toml` currently has no `[project.scripts]`. Add:

```toml
[project.scripts]
ar-harness = "harness:main"
```

Without this, `uv run ar-harness` fails on the remote. This is a prerequisite.

---

## Step 1: `skills/ar-design/references/program-protocol.md`

Extract the **fixed protocol sections** from `/Users/tmwsiy/code/autoresearch/program.md` — the invariant parts that apply to any experiment:
- Branch creation (step 1-6 of Setup)
- Experiment loop structure (LOOP FOREVER block)
- Output format (the `---` summary block)
- TSV logging format (columns, example)
- Never-stop rule
- Simplicity criterion
- Timeout and crash handling rules

**Omit** problem-specific content: the GPT/climbmix goal, the specific metric name, the specific grep pattern, the time budget value, the files section. Those get injected per-experiment by ar-design.

Use placeholder tokens like `{METRIC_NAME}`, `{METRIC_DIRECTION}`, `{TIME_BUDGET}`, `{RUN_COMMAND}`, `{GREP_PATTERN}` where ar-design will substitute values.

---

## Step 2: `skills/ar-design/SKILL.md`

### Frontmatter
```yaml
---
name: ar-design
description: >
  Generate complete experiment directories for the autoresearch protocol — program.md,
  train.py, prepare.py, pyproject.toml. USE WHEN designing an autoresearch experiment,
  scaffolding a new AR run, generating experiment files, creating a train.py baseline.
argument-hint: ["problem description", "prepare.py=/path/to/existing.py <problem>"]
allowed-tools: Read, Write, Bash(mkdir:*), Bash(git:*)
---
```

### Content structure

**Critical Rules** (lead with these):
1. **Human gate is inviolable.** Never write program.md or train.py until the user explicitly approves prepare.py. Wrong metric = wasted GPU hours.
2. prepare.py defines the metric contract — it must be deterministic, correct, and leak-free (no val data in train).
3. The protocol template lives at `references/program-protocol.md` — read it and inject values, never rewrite the protocol from scratch.

**Steps:**
1. Parse `$ARGUMENTS`. Detect `prepare.py=<path>` prefix (bring-your-own path).
2. Infer experiment name: `<domain>-<dataset>-<date>` (e.g. `gpt-shakespeare-apr04`).
3. `mkdir -p <name>`, `git init`, make initial commit.
4. **If user provided prepare.py**: copy it into the experiment dir, read to extract metric name, constants, exported interface. Skip gate.
5. **Otherwise**: generate prepare.py candidate. Present to user with a summary:
   - What metric it computes and how
   - What dataset and how it's split (train vs val)
   - What constants are fixed (MAX_SEQ_LEN, TIME_BUDGET, etc.)
   - What interface train.py imports from it
   
   **STOP and wait for approval.** User can: approve, request changes (regenerate), or edit manually. Do not proceed until explicit approval.

6. Read `references/program-protocol.md`. Fill in placeholders:
   - `{METRIC_NAME}` — e.g. `val_bpb`
   - `{METRIC_DIRECTION}` — `lower is better` or `higher is better`
   - `{TIME_BUDGET}` — seconds
   - `{RUN_COMMAND}` — `uv run train.py`
   - `{GREP_PATTERN}` — e.g. `^val_bpb:`
   
   Add a **Problem** section at the top of program.md describing the goal in concrete terms.
   Add a **Files** section: what the agent can/cannot modify.
   Add **Hardware** context if provided.
   Write `program.md`.

7. Generate `train.py` scaffold. Use domain templates for known domains:
   - **NLP pretraining (GPT)**: use `/Users/tmwsiy/code/autoresearch/train.py` as the template — copy it and note what constants to adjust. For non-climbmix datasets, note the prepare.py interface changes needed.
   - **Character-level LM**: small GPT (4-8 layers, 128-256 dim), AdamW + cosine LR schedule, char tokenizer
   - **Vision classification**: standard torchvision ViT or ResNet baseline, top-1 accuracy metric, CrossEntropyLoss
   - **Fine-tuning**: HuggingFace transformers PEFT baseline
   - **Novel/unknown domain**: generate best-effort baseline from scratch, add comment block: `# NOTE: This scaffold was generated for an unfamiliar domain. Review before running.`

8. Generate `pyproject.toml`:

```toml
[project]
name = "<experiment-name>"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ar-harness",
    # problem-specific deps below
]

[tool.uv.sources]
ar-harness = { path = "../ar-harness" }
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

   Note: `../ar-harness` is relative — resolves correctly when ar-run places experiment and ar-harness both under `~/experiments/`.

9. Write all files. Git commit: `"ar-design: initial scaffold"`.

10. Print a summary: experiment dir path, files created, metric being optimized, next step (`/ar-run <dir> eddie@tela`).

---

## Step 3: `skills/ar-run/SKILL.md`

### Frontmatter
```yaml
---
name: ar-run
description: >
  Deploy and manage autoresearch experiments on remote GPU machines via SSH + tmux.
  USE WHEN running an autoresearch experiment, deploying to remote GPU, checking
  experiment status, collecting results, stopping a remote run.
argument-hint: ["<experiment-dir> <ssh-target>", "status <ssh-target>", "tail <target> <session>", "stop <target> <session>", "collect <target> <session>"]
allowed-tools: Bash(ssh:*), Bash(rsync:*), Read
disable-model-invocation: true
---
```

### Content structure

**Critical Security Rule** (lead with this):
- `CLAUDE_OAUTH_TOKEN` must be injected as an env var **inside the tmux command string only**.
- Never pass as a CLI arg (visible in `ps`).
- Never echo it or let it appear in `run.log`.
- Read it from the local shell: `$CLAUDE_OAUTH_TOKEN`.
- The env var prefix inside the tmux command string is safe — tmux does not log command args.

**Subcommand dispatch** (parse first word of `$ARGUMENTS`):

| First word | Action |
|------------|--------|
| (path) | Deploy and launch |
| `status` | List active sessions + tail last log line from each |
| `tail` | `tmux capture-pane` last 50 lines |
| `stop` | Send C-c, wait for graceful exit |
| `collect` | rsync results back + print summary |

**Deploy operation (default):**

```bash
NAME=$(basename <experiment-dir>)
TARGET=<ssh-target>
REMOTE_BASE=~/experiments

# 1. Validate remote prerequisites
ssh $TARGET "which uv && which git && which tmux && git config user.name && git config user.email" \
  || { echo "ERROR: Missing remote prerequisites. Install uv, git, tmux and set git user config."; exit 1; }

# 2. Validate local experiment dir
# Check for: program.md, train.py, prepare.py, pyproject.toml

# 3. rsync experiment + ar-harness
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  <experiment-dir>/ $TARGET:$REMOTE_BASE/$NAME/
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  /Users/tmwsiy/code/ar-harness/ $TARGET:$REMOTE_BASE/ar-harness/

# 4. SSH chain: sync deps, run prepare.py, launch tmux
ssh $TARGET "
  cd $REMOTE_BASE/$NAME &&
  uv sync --quiet &&
  echo '[ar-run] Running prepare.py (may take minutes if downloading data)...' &&
  uv run python prepare.py &&
  tmux new-session -d -s $NAME \
    'CLAUDE_OAUTH_TOKEN=$CLAUDE_OAUTH_TOKEN uv run ar-harness --cwd . --max-turns 100 --model claude-sonnet-4-6 2>&1 | tee run.log'
"

# 5. Print connection info
echo "Launched: $NAME"
echo "Attach:   ssh $TARGET -t tmux attach -t $NAME"
echo "Tail:     /ar-run tail $TARGET $NAME"
```

**status:**
```bash
ssh <target> "
  tmux list-sessions 2>/dev/null || echo '(no active sessions)';
  echo '---';
  for f in ~/experiments/*/run.log; do
    echo \$f;
    tail -1 \$f 2>/dev/null;
    echo;
  done
"
```

**tail:**
```bash
ssh <target> "tmux capture-pane -t <session> -p -S -50"
```

**stop:**
```bash
ssh <target> "tmux send-keys -t <session> C-c"
# Wait up to 30s, then offer force kill:
# ssh <target> "tmux kill-session -t <session>"
```

**collect:**
```bash
rsync -avz <target>:~/experiments/<session>/results.tsv <local-experiment-dir>/
rsync -avz <target>:~/experiments/<session>/run.log <local-experiment-dir>/
# Then parse results.tsv and print:
# - Best result (lowest/highest metric)
# - Total runs: N (keep: X, discard: Y, crash: Z)
```

**Additional notes:**
- Session name = `basename` of experiment dir, sanitized to `[a-z0-9-]` (replace underscores and other chars with hyphens)
- `prepare.py` runs unconditionally — it self-skips if data cache already exists
- Default model: `claude-sonnet-4-6`, default max-turns: 100
- For `collect`, print the results.tsv summary table after syncing

---

## Verification

1. **ar-harness entry point**: `cd /Users/tmwsiy/code/ar-harness && uv run ar-harness --help` should print usage.

2. **ar-design smoke test**: `/ar-design Pretrain a character-level LM on Shakespeare, minimize val_bpb, single GPU, 3 minute budget`
   - Gate should fire — agent presents prepare.py and waits
   - After approval, all 4 files should appear in the experiment dir
   - pyproject.toml should have `ar-harness = { path = "../ar-harness" }` in `[tool.uv.sources]`

3. **ar-run deploy**: `/ar-run ~/experiments/gpt-shakespeare-test eddie@tela`
   - Prereq check runs (uv/git/tmux/git-config)
   - Both dirs rsynced
   - tmux session visible via `ssh eddie@tela tmux list-sessions`
   - `run.log` should NOT contain the OAuth token

4. **ar-run status**: `/ar-run status eddie@tela` — lists sessions and last log line

5. **ar-run collect**: `/ar-run collect eddie@tela gpt-shakespeare-test` — pulls results.tsv and run.log, prints summary

---

## After creating the skill files

Install locally to activate:
```bash
cd /Users/tmwsiy/code/toolshed
./toolshed.sh install ar-design --global
./toolshed.sh install ar-run --global
```

This copies them to `~/.claude/skills/` (the `claude` target in `toolshed.yaml`). After install, `/ar-design` and `/ar-run` are available in Claude Code immediately.

To deploy to trinity/OpenClaw later, add both to `hosts.yml` and run the Ansible deploy playbook.

---

## Key context

- Skills live in `/Users/tmwsiy/code/toolshed/skills/<name>/SKILL.md` — read 2-3 existing ones (e.g. `notify/`, `jenkins-api/`, `infra/`) before writing to learn the conventions
- `disable-model-invocation: true` in ar-run frontmatter prevents accidental triggering (it's a side-effectful deploy operation)
- ar-design does NOT use `disable-model-invocation` — it's conversational (must wait for human gate approval)
- The `references/program-protocol.md` file is read at runtime by ar-design using the Read tool — it's not embedded in SKILL.md
- OAuth token note: the claude-agent-sdk uses `CLAUDE_OAUTH_TOKEN` (not `ANTHROPIC_API_KEY`)
