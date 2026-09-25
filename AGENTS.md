# AI Assistant Fine-Tuning — Agent Instructions


This document helps AI coding agents understand the command flow, data structures, and processing patterns in this fine-tuning pipeline.

## Project Overview

The **AI Assistant Fine-Tuning** project builds training data for a Serbian
teaching assistant from published Petlja course material, then fine-tunes
`Qwen/Qwen3-14B` on it with LoRA:

1. **gen-questions** — one question per segment of each teaching activity
2. **gen-answers** — the local model answers its own questions
3. **judge** — the teacher model scores each answer, then repairs or rewrites it

Step 3 stops at a **judged file** carrying the question, the model's answer and
the teacher's. `make-sft` and `make-dpo` turn that into training files, and they
cost nothing — so the shape of a training run is never welded to the pass that
cost hours of teacher calls.

---

## Command Flow & Data I/O

The pipeline is **7 CLI commands** (via the `astft` entrypoint, registered in
[cli.py](ai_assistant_fine_tuning/cli.py)): three that cost API calls
(`gen-questions`, `gen-answers`, `judge`), two that are free and repeatable
(`make-sft`, `make-dpo`), and two that train (`sft-finetune`, `dpo-finetune`).

### **THE PIPELINE**

Steps 1-3 cost API calls and run once. `make-sft`/`make-dpo` cost nothing and
are re-run per training run, in whatever shape it needs.

#### **Step 1: gen-questions**
- **Purpose:** Generate one question per segment of a course's teaching activities
- **Models:** Two passes — pass 1 on the local model (`--local-model`, default
  `Qwen/Qwen3.8-27B-FP8`, free), pass 2 on the teacher model (`--model`, default
  `gpt-5.4-mini`, Azure when `CHATAI_AZURE_API_KEY` is set)
- **Input:**
  - `--input-dir` (default: `../data-preprocessing/dataset/`) — one course
    directory with a `structure.json`, or a dataset root of them. Loose `*.md`
    with no structure.json is no longer enough: step 1 works per lecture.
- **Output:** two files, identical shape, one JSONL line per question, differing
  only in what fills the `--- ГРАДИВО ---` block of `system_message`
  - `questions_derived.jsonl` — the per-question context pass 2 wrote
  - `questions_full.jsonl` — the whole activity
    ```json
    {
      "source": "1_lekcija/1_1_cas.md",
      "course": "os6_inf_prog",
      "activity": "1.1",
      "segment_kind": "exposition|task",
      "system_message": "--- О ЛЕКЦИЈИ --- arc ... --- ГРАДИВО --- ...",
      "questions": ["one question"]
    }
    ```
- **Options:** `--context both|derived|full`, `--model`, `--concurrency` (pass 2,
  default 4), `--local-model`, `--local-concurrency` (pass 1, default 4),
  `--local-base-url`, `--local-api-key`, `--azure-endpoint`, `--azure-api-version`,
  `--skip-practice/--include-practice`, `--output-dir`, `--append/--overwrite`
- **Key File:** [gen_questions.py](ai_assistant_fine_tuning/gen_questions.py)

> **Pass 1 must run greedy.** The `teaching`/`practice` label is a
> classification. At `temperature=0.7` two identical runs of `os1_dig_svet` gave
> 24 and 11 teaching activities — half the corpus appeared and disappeared.
>
> **Lecture and activity numbers are positional** counts over `structure.json`,
> not parsed out of filenames: only 6 of the 35 courses in the dataset have two
> numbers in every filename, and only 2 keep their lectures in `<n>_lekcija`
> directories. So `1.4` is "the 4th non-quiz activity of the 1st lecture", which
> need not match the course's own label — os6 calls that one 1.5, because its
> quiz took 1.4. The label the course uses is in `title`.
>
> **Concurrency is API-calls-only.** Both passes fan out through a
> `ThreadPoolExecutor`, but every file write happens on the main thread as
> results arrive, so there is no shared writer and no lock. Records land in
> completion order, not document order — join by `source` + question, never by
> line number. `--concurrency` defaults to 4 because the Azure deployment is
> shared; retryable rejections back off 4/8/16/32/64s before the activity is
> given up. Measured on `os1_dig_svet`: 333s serial → 111s at 4-way.
>
> **A course that maps zero lectures aborts the run** before pass 2. That is
> what a dead SSH tunnel to the vLLM box looks like, and the old behaviour —
> two empty files and exit 0 — would silently waste an unattended loop.
>
> **No answer is generated in step 1.** Answering belongs to step 2, against
> whichever context that run trains on. Writing the answer in step 1 would write
> it from the full activity while the derived run only ever shows the model a
> paragraph — training it to assert things its context does not support. Run
> `gen-answers` once per output file to get two answer sets.

#### **Step 2: gen-answers**
- **Purpose:** Answer questions using a local model (vLLM) — **usually runs once, output reused**
- **Model:** Local model served at `--base-url` (default: `http://localhost:8000/v1`)
- **Input:**
  - `--questions FILE` (default: `questions_derived.jsonl`) — From Step 1; run once per context variant
- **Output:**
  - `answers.jsonl` — One JSONL line per question
    ```json
    {
      "source": "relative/path/file.md",
      "system_message": "...",
      "question": "Q1",
      "answer": "Model-generated answer..."
    }
    ```
- **Options:** `--model`, `--output`, `--base-url`, `--api-key`, `--concurrency`,
  `--append/--overwrite`
- **Key File:** [gen_answers.py](ai_assistant_fine_tuning/gen_answers.py)

> **The answer is stripped before it is written.** A vLLM server running
> `--reasoning-parser` returns the content with the blank lines where the thinking
> block was, and this string ships into the training records verbatim — leading
> whitespace in an assistant turn is something the model would learn to
> reproduce.
>
> **The `answer` field is the model's own attempt, and it stays that way.** Step 3
> keeps it beside the teacher's answer rather than replacing it, because it is the
> `rejected` side of every DPO pair. Nothing downstream should overwrite it.

#### **Step 3: judge** (score, then improve)
- **Purpose:** Score every answer, then have the teacher improve it. **Writes no
  training data** — it stops at the judged file. `make-sft`/`make-dpo` build
  training files from that, for free.
- **Model:** Teacher model (`--model`, default `gpt-5.6-sol`)
- **Input:** `--answers FILE` (default: `answers.jsonl`) — from step 2
- **Output:**
  - `scores.jsonl` (`--scores-output`) — pass 1, every answer with both scores
  - `written.jsonl` (`--written-output`) — pass 2, one row per question:
    ```json
    {
      "source": "1_lekcija/1_1_cas.md",
      "system_message": "--- О ЛЕКЦИЈИ --- ... --- ГРАДИВО --- ...",
      "question": "one question",
      "answer": "what the local model said",
      "improved_answer": "what the teacher said",
      "mode": "repair|rewrite",
      "substance_score": 82,
      "language_score": 91,
      "judge": "two-pass-rank"
    }
    ```
- **Options:** `--write repair|rewrite|split`, `--rewrite-share` (default 0.25,
  `split` only), `--model`, `--prompts`, `--concurrency` (default 4),
  `--resume/--no-resume`, `--scores-only`, `--limit`
- **Key Files:** [judge.py](ai_assistant_fine_tuning/judge.py) (command),
  [judging.py](ai_assistant_fine_tuning/judging.py) (prompts, scorer, record builders)

> **`--write` is the mode of the teacher's answer, not a routing decision.** It
> says nothing about SFT or DPO: either training file can be built from either
> kind of row. What it changes is how far the teacher's answer travels from the
> model's — `repair` deletes and corrects while keeping the author's words,
> `rewrite` starts from the material and never sees the attempt. That distance is
> what a DPO pair is made of.
>
> **The split is by rank, never by a threshold.** A rank cannot swing: the worst
> quarter is the worst quarter whatever the model does with the scale, and the
> scale does move — the same answers scored 48-94 under one revision of the
> prompt and 43-95 under the next. `--rewrite-share` is applied *after* scoring,
> so a batch that has been paid for can be re-split at a different share without
> being re-scored.
>
> **`--prompts FILE` swaps in a different rubric** without editing the
> built-in prompts, which would change what every result filed under
> `JUDGE_NAME` means. The file is TOML: `name`, `score_prompt`, `repair_prompt`,
> `rewrite_prompt`, and optionally `system_judge` / `system_write`. Use `'''`
> literal strings. The name must differ from the built-in `two-pass-rank`.
> Placeholders are checked before anything is paid for: score and repair take
> exactly `{system_message}`, `{question}`, `{answer}`, and rewrite exactly
> `{system_message}`, `{question}`, since it never sees the attempt. Write a
> literal brace as `{{` or `}}`. The score prompt must still
> ask for `substance_score` and `language_score`, since that response format is
> fixed.
>
> **Every scored row records its `judge`.** `--resume` refuses output files
> holding rows from a different judge: two rubrics' scores are not on one
> scale, and the split would rank them together. Rows from before the field
> existed count as `two-pass-rank`.
>
> **Scores are 1-100** (`substance_score` and `language_score`, kept apart).
> Anything produced before 2026-08-18 is on the old 1-10 scale and is **not
> comparable** — do not mix old and new score files.
>
> **The judged file is the expensive artefact.** Pass 2 costs hours of teacher
> calls; the shape of a training file costs nothing. Never delete
> `written_*.jsonl` — every training file is rebuilt from it in under a minute,
> so a training file can be thrown away freely.

#### **make-sft** (judged file → SFT training data)
- **Purpose:** Teacher's answer as the target; the model's answer is ignored.
  No API calls.
- **Input:** `--written FILE` (default: `written.jsonl`)
- **Output:** `--output` (default: `training_data.jsonl`), in one of two shapes:
    ```json
    {"prompt": [system, user], "completion": [assistant]}
    {"messages": [system, user, assistant]}
    ```
- **Options:** `--rows all|repair|rewrite`, `--shape prompt-completion|messages`,
  `--max-tokens`, `--tokenizer-model`
- **Key File:** [make_td.py](ai_assistant_fine_tuning/make_td.py)

> **`--shape` decides what the loss covers, and it is the single highest-leverage
> flag in this repo.** `prompt-completion` makes TRL resolve
> `completion_only_loss` to True — loss on the answer tokens only. `messages` is
> a language-modelling dataset: loss over the whole sequence, lesson material
> included. The material is handed to the model at inference and never generated,
> and it is ~77% of the tokens (mean 472 answer tokens of 2029). Measured on
> belebele-sr against base: `t7-sft` (whole-sequence loss, April data) lost
> **4.4 points**; `derived-sft-02` (`prompt-completion`, September data) lost
> **0.8**. That comparison changes the data as well as the loss, so it is
> suggestive, not controlled — `derived-sft-01`, the September `messages` run,
> was never evaluated. The two shapes render byte-identically once
> concatenated; only the scored tokens differ.
>
> `assistant_only_loss=True` is **not** an alternative — Qwen3's chat template
> has no `{% generation %}` marker, so the assistant mask comes back all zeros.

#### **make-dpo** (judged file → DPO pairs)
- **Purpose:** `chosen` = teacher's answer, `rejected` = model's answer. No API calls.
- **Input:** `--written FILE` (default: `written.jsonl`)
- **Output:** `--output` (default: `training_data_dpo.jsonl`), conversational so
  TRL applies the chat template:
    ```json
    {
      "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "Q"}],
      "chosen": [{"role": "assistant", "content": "the teacher's answer"}],
      "rejected": [{"role": "assistant", "content": "the model's answer"}]
    }
    ```
- **Options:** `--rows all|repair|rewrite`, `--max-tokens` (measured on the
  longer side), `--tokenizer-model`
- **Key File:** [make_td.py](ai_assistant_fine_tuning/make_td.py)

> **Prefer `--rows repair`.** A repair pair differs by deletions and corrections
> only — same words, no length gap — so the model has to learn something to tell
> the sides apart. A rewrite pair is two unrelated texts whose rejected side
> averages ~4.5× longer. Measured on rewrite pairs alone: loss **0.0000 by step
> 40 of 219**, reward accuracy 1.00 from step 20, gradient norms ~1e-5 for the
> rest of the run. It learnt nothing after step 40, and part of what it did learn
> was "be brief".
>
> Pairs whose sides are identical are dropped: they put the loss at log 2 with a
> zero gradient.
>
> **Conversational, not strings.** An earlier version emitted `prompt` as a raw
> concatenated string. TRL reads that as its *standard* format and tokenises it
> verbatim — training on a surface form neither SFT nor vLLM ever produces, since
> both go through `apply_chat_template`.

---

### **FINE-TUNING (Training)**

Run **one or the other**, not one chained into the other. See the warning below.

#### **sft-finetune**
- **Purpose:** LoRA SFT on a single CUDA GPU
- **Input:** `--data FILE` (default: `training_data.jsonl`) — from `make-sft`
- **Output:** adapter + one checkpoint per epoch in `--output-dir`
- **Key File:** [finetune_sft.py](ai_assistant_fine_tuning/finetune_sft.py)
- Always creates a **fresh** LoRA; it cannot continue from an existing adapter.

#### **dpo-finetune**
- **Purpose:** LoRA DPO continuing from an existing SFT adapter
- **Input:** `--adapter DIR` (required) + `--training-data FILE` — from `make-dpo`
- **Output:** a **standalone** adapter in `--output-dir`. It carries the same LoRA
  tensors trained further, so serve it *or* the SFT adapter against the base —
  never both stacked. The `--adapter` directory is only read, never written.
- **Key File:** [finetune_dpo.py](ai_assistant_fine_tuning/finetune_dpo.py)

> **`--max-seq-len` must come from the data.** Both scripts default to 2048.
> Measured against the real chat template that truncates 41.5% of the derived set
> and ~90% of the full set — and truncation removes the *end* of the sequence,
> which is the answer. Keep it equal to the `make-sft --max-tokens` used to build
> the file.
>
> **Do not chain SFT → DPO by reflex.** With `rejected` coming from the *pre-SFT*
> model, a freshly trained adapter already separates the pairs at initialisation:
> the objective is satisfied in a few dozen steps and the rest is near-zero
> gradient. If DPO is wanted after SFT, its rejected side should come from the SFT
> model itself — re-run `gen-answers` against the adapter, then `judge` again — or
> at minimum from `make-dpo --rows repair`.

---

## Key Data Structures

### JSONL Records

**questions_derived.jsonl** / **questions_full.jsonl** (Step 1 output):
```json
{
  "source": "str — path relative to --input-dir",
  "course": "str — course directory name",
  "activity": "str — positional <lecture>.<activity>",
  "segment_kind": "exposition|task",
  "system_message": "str — lecture arc + context block",
  "questions": ["exactly one question"]
}
```

**answers.jsonl** (Step 2 output):
```json
{
  "source": "str",
  "system_message": "str",
  "question": "str",
  "answer": "str — model-generated answer"
}
```

**scores.jsonl** (`judge` pass 1) — every answers.jsonl field, plus:
```json
{
  "substance_score": 82,
  "language_score": 91,
  "judge": "str — the name of the judge that scored it"
}
```

**written.jsonl** (`judge` pass 2 — the judged file, and the expensive artefact):
```json
{
  "source": "str",
  "system_message": "str",
  "question": "str",
  "answer": "str — what the local model said",
  "improved_answer": "str — what the teacher said",
  "mode": "repair|rewrite",
  "substance_score": 82,
  "language_score": 91,
  "judge": "str — the name of the judge that scored it"
}
```
Self-contained on purpose: this row alone becomes an SFT row or a DPO pair, with
no lookup back into the scores file. Rows land in **completion order**, not
document order, because both passes fan out — join by `source` + `question`,
never by line number.

**training_data.jsonl** (`make-sft`), default shape `prompt-completion`:
```json
{
  "prompt": [
    {"role": "system", "content": "str"},
    {"role": "user", "content": "str"}
  ],
  "completion": [{"role": "assistant", "content": "str — the teacher's answer"}]
}
```
With `--shape messages` instead:
```json
{
  "messages": [
    {"role": "system", "content": "str"},
    {"role": "user", "content": "str"},
    {"role": "assistant", "content": "str — the teacher's answer"}
  ]
}
```
The two render byte-identically once concatenated. What differs is which tokens
the loss covers — see the `make-sft` warning above.

**training_data_dpo.jsonl** (`make-dpo`) — message **lists** on every side, so TRL
treats it as conversational and applies the chat template:
```json
{
  "prompt": [
    {"role": "system", "content": "str"},
    {"role": "user", "content": "str"}
  ],
  "chosen": [{"role": "assistant", "content": "str — the teacher's answer"}],
  "rejected": [{"role": "assistant", "content": "str — the model's own answer"}]
}
```

---

## Default Paths & Conventions

| Variable | Default | Purpose |
|----------|---------|---------|
| Input dataset | `../data-preprocessing/dataset/` | Markdown lecture files (etit1_prog, gim1_rac_ikt, etc.) |
| Questions output | `finetunedata/questions/` | Step 1 output (two context variants) |
| Answers input | `answers.jsonl` | Step 2 input/output |
| Scores | `scores.jsonl` | Step 3 pass 1 |
| Judged file | `written.jsonl` | Step 3 pass 2 — **the expensive artefact, keep it** |
| Training data (SFT) | `training_data.jsonl` | `make-sft` output |
| Training data (DPO) | `training_data_dpo.jsonl` | `make-dpo` output |
| Fine-tune output | `finetune-output/<context>-<stage>-<NN>/` | Adapters + checkpoints |

Run directories are named `<context>-<stage>-<NN>`: `context` is the question
context variant (`derived` or `full`), `stage` is `sft` or `dpo`, `NN` bumps when
any knob changes — e.g. `full-sft-01`. Older names `t7-sft`/`t7-dpo` (a 7/10 score
split) and `v1` (a smoke test) predate this.

---

## Common Workflows

### **Workflow A: one course, end to end**
```bash
# Step 1 — questions, two passes. Pass 1 needs the local model on port 8000.
astft gen-questions --input-dir ./dataset/os6_inf_prog --output-dir finetunedata/questions/

# Step 2 — the local model answers. Once per context variant: that is the experiment.
astft gen-answers --questions finetunedata/questions/questions_full.jsonl \n    --output finetunedata/answers/answers_full.jsonl

# Step 3 — score, then repair or rewrite. Hours. Resumable. Writes no training data.
astft judge --answers finetunedata/answers/answers_full.jsonl \n    --scores-output finetunedata/td/scores_full.jsonl \n    --written-output finetunedata/td/written_full.jsonl \n    --write split --rewrite-share 0.25

# Build a training file (free, repeatable), then train on it
astft make-sft --written finetunedata/td/written_full.jsonl \n    -o finetunedata/td/sft_full.jsonl --rows all --max-tokens 8192
astft sft-finetune --data finetunedata/td/sft_full.jsonl \n    --output-dir ./finetune-output/full-sft-01 \n    --max-seq-len 8192 --epochs 2 --batch-size 1 --grad-accum 16
```

### **Workflow B: another run from a judged file already paid for**
No API calls. This is the loop that should run many times per judged file.
```bash
# Answer-only loss, both halves fused into one SFT set
astft make-sft --written finetunedata/td/written_full.jsonl \n    -o finetunedata/td/sft_fused.jsonl --rows all --shape prompt-completion

# Or: hard DPO pairs from the repaired rows only
astft make-dpo --written finetunedata/td/written_full.jsonl \n    -o finetunedata/td/dpo_repair.jsonl --rows repair --max-tokens 8192
astft dpo-finetune --adapter ./finetune-output/full-sft-01 \n    --training-data finetunedata/td/dpo_repair.jsonl \n    --output-dir ./finetune-output/full-dpo-01 --max-seq-len 8192
```

### **Workflow C: re-split a batch without paying again**
`--rewrite-share` is applied after scoring, so changing the mix costs one pass-2
re-run of only the records whose mode changed — `--resume` keys the judged file by
`(record, mode)` and keeps the rest.
```bash
astft judge --answers finetunedata/answers/answers_full.jsonl \n    --scores-output finetunedata/td/scores_full.jsonl \n    --written-output finetunedata/td/written_full.jsonl \n    --write split --rewrite-share 0.40
```

---

## Development Notes

### Module Organization
- **[cli.py](ai_assistant_fine_tuning/cli.py)** — Click CLI group and command registration
- **[gen_questions.py](ai_assistant_fine_tuning/gen_questions.py)** — Step 1 (sync)
- **[gen_answers.py](ai_assistant_fine_tuning/gen_answers.py)** — Step 2 (sync)
- **[judge.py](ai_assistant_fine_tuning/judge.py)** — Step 3 command (two passes)
- **[judging.py](ai_assistant_fine_tuning/judging.py)** — Step 3 prompts, scorer, record builders (leaf, no CLI)
- **[make_td.py](ai_assistant_fine_tuning/make_td.py)** — `make-sft` and `make-dpo`, no API calls
- **[io_jsonl.py](ai_assistant_fine_tuning/io_jsonl.py)** — JSONL read/write (atomic whole-file writes)
- **[core.py](ai_assistant_fine_tuning/core.py)** — Shared logic (clients, markdown collection, defaults)
- **[finetune_sft.py](ai_assistant_fine_tuning/finetune_sft.py)** — SFT fine-tuning (Hugging Face Trainer)
- **[finetune_dpo.py](ai_assistant_fine_tuning/finetune_dpo.py)** — DPO fine-tuning (TRL)

### Configuration
- Teacher model: `gpt-5.6-sol` (Azure when `CHATAI_AZURE_API_KEY` is set, else OpenAI)
- Local model: `Qwen/Qwen3-14B` (answering, and the base for fine-tuning)
- Scores: 1–100, `substance_score` and `language_score`, kept apart
- `--rewrite-share` default 0.25 — a **rank**, not a threshold, applied after scoring

### The Batch API path is gone
`gen-batch`, `batch-submit` and `fetch-batch` were removed along with
`batch.py`, `batch_util.py` and `gen_batch.py`. Step 1 never had a batch form
(two dependent passes), and step 3's resume plus checkpointing covers what the
batch path was for. Do not reintroduce references to them.

### Virtual Environment Setup

**IMPORTANT:** The virtual environment has special path considerations:

1. **Location:** The `.venv` directory is located at the project root, but the actual Python environment may be symlinked to `/datadisk/projects/AI-Assistant-Fine-Tuning/.venv/`

2. **Running Commands:**
   - **Always activate the venv first:** `source .venv/bin/activate`
   - After activation, use commands directly: `astft`, `python`, etc.
   - If pip is not available, install it: `python -m ensurepip --upgrade`
   - Then use: `python -m pip` for package management

3. **Common Issues:**
   - **Direct pip calls fail:** `.venv/bin/pip` may not exist initially
   - **Solution:** Use `python -m pip` after activating the venv
   - **Path confusion:** The workspace at `/home/azureuser/projects/` is actually symlinked to `/datadisk/projects/`
   - Activation handles this automatically, so always use `source .venv/bin/activate`

4. **Dependency Management:**
   - Project uses `pyproject.toml` for dependency specification
   - If you encounter dependency warnings (e.g., urllib3/requests compatibility):
     - Ensure pip is available: `python -m ensurepip --upgrade`
     - Upgrade problematic packages: `python -m pip install --upgrade <package>`

5. **Example Session:**
   ```bash
   # Activate venv (handles path symlinks automatically)
   source .venv/bin/activate
   
   # Ensure pip is available (if needed)
   python -m ensurepip --upgrade
   
   # Install/upgrade packages
   python -m pip install --upgrade requests
   
   # Run commands (astft is now in PATH)
   astft judge --answers answers.jsonl
   ```

### Error Handling & Retry
- Retryable rejections (429, timeout, connection error) back off 4/8/16/32/64s
  before the record is given up and logged
- `judge` checkpoints both output files every 250 records; `--resume` reuses them
- A failed record is skipped, not fatal: an unattended pass finishes and reports
  its failure count

---

## When Adding Features

- **New command?** Add to [cli.py](ai_assistant_fine_tuning/cli.py) and define input/output clearly
- **New training-file shape?** It belongs in [make_td.py](ai_assistant_fine_tuning/make_td.py),
  derived from the judged file. Never add a shape that requires re-running `judge`
- **Changing what a record looks like?** Change the builders in
  [judging.py](ai_assistant_fine_tuning/judging.py) and rebuild from
  `written_*.jsonl` — no teacher calls needed
- **Add LLM calls?** Keep clients in [core.py](ai_assistant_fine_tuning/core.py)
  and prompts in [judging.py](ai_assistant_fine_tuning/judging.py)
- **Change file paths?** Update defaults in constants (e.g., `DEFAULT_SFT_OUTPUT_FILE`)
