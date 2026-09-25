# AI-Assistant-Fine-Tuning


## Running Commands

You can run commands in either of these ways:

1. Installed script entrypoint:

```bash
astft <command> [options]
```

## Top-Level CLI

```bash
astft --help
astft --version
```

## Commands


### 1) gen-questions

**Step 1** — Generate one question per segment of a course's teaching activities,
in two passes.

**Pass 1 — local model, free.** One call per lecture, all its activities at once.
Returns the lecture `arc` and a `teaching`/`practice` label per activity. Both
need a lecture-wide view: "does this introduce material, or exercise material
introduced earlier" cannot be answered one activity at a time, and the arc is by
definition about the whole lecture. It runs greedy — the label is a
classification, and at `temperature=0.7` two identical runs of `os1_dig_svet`
gave 24 and 11 teaching activities.

**Pass 2 — teacher model.** One call per teaching activity. Cuts it into
consecutive segments and writes one question per segment, plus a written
`context` per segment: a self-contained explanation of what that question
targets, built from the passage the segment covers but not copied from it. Pass
1's arc rides along as input so the activity can be placed in its lecture.

**No answer is generated.** Answering belongs to `gen-answers`, which sends
`system_message` + question and nothing else — so running it once per output file
gives two answer sets, each derivable from its own context.

Quizzes never appear: `structure.json` tags them `source_type: "quiz"`.

```bash
astft gen-questions --help
```

Options:

- `--input-dir DIRECTORY` (default: `../data-preprocessing/dataset/`) — one course
  directory with a `structure.json`, or a dataset root of them
- `--output-dir, -o DIRECTORY` (default: `finetunedata/questions/`)
- `--context [both|derived|full]` (default: `both`)
- `--model, -m TEXT` (default: `gpt-5.4-mini`) — pass 2
- `--concurrency INT` (default: `4`) — parallel pass-2 requests
- `--local-model TEXT` (default: `Qwen/Qwen3.8-27B-FP8`) — pass 1
- `--local-concurrency INT` (default: `4`) — parallel pass-1 requests
- `--local-base-url TEXT` (default: `http://localhost:8000/v1`)
- `--local-api-key TEXT` (default: `EMPTY`)
- `--azure-endpoint TEXT` (default: `$CHATAI_AZURE_ENDPOINT`)
- `--azure-api-version TEXT` (default: `2024-10-21`)
- `--skip-practice / --include-practice` (default: skip-practice)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft gen-questions --input-dir ./dataset/os6_inf_prog --output-dir finetunedata/questions/
```

Pass 1 needs the local model on port 8000. If a course maps *no* lecture at all,
the command stops before pass 2 rather than writing empty files — that is what a
dead SSH tunnel looks like, and it must not burn teacher-model calls in an
unattended loop. The teacher model goes through Azure when `CHATAI_AZURE_API_KEY`
is set (the gpt-5.x deployments live there), and through `OpenAI()` otherwise.

Both passes run their API calls concurrently while every file write stays on the
main thread, so there is no shared writer. Records land in **completion order**,
not document order: JSONL lines are independent and joins are by `source` +
question, never by line number. `--concurrency` stays deliberately mild because
the Azure deployment is shared; a rejection (429, timeout, connection error)
backs off 4s, 8s, 16s, 32s, 64s before the activity is given up and logged.

Measured on `os1_dig_svet` (4 lectures, 14 teaching activities): 333s serial,
111s at `--concurrency 4 --local-concurrency 4`.

Output files (inside output dir) — identical shape, one record per question,
differing only in what fills the `--- ГРАДИВО ---` block of `system_message`:

- `questions_derived.jsonl` — the written per-question context
- `questions_full.jsonl` — the whole activity

```
--- О ЛЕКЦИЈИ ---          arc, per lecture, from pass 1
--- О АКТИВНОСТИ ---       activity_context, per activity, from pass 2   (derived only)
--- ГРАДИВО ЗА ПИТАЊЕ ---  that question's context, or the whole activity
--- КРАЈ ГРАДИВА ---
```

`activity_context` is not in the full variant: the whole activity follows two
lines below it, so a written description of it would only repeat what the reader
is about to read. That does mean the two files differ in two ways rather than one
— if derived wins, that is a confound to resolve.

Example JSONL:
```json
{"source": "1_lekcija/1_1_cas.md", "course": "os6_inf_prog", "activity": "1.1", "segment_kind": "exposition", "system_message": "--- О ЛЕКЦИЈИ --- ...", "questions": ["..."]}
```

`source`, `system_message` and `questions` are what `gen-answers` reads; `course`,
`activity` and `segment_kind` ride along for sampling and are ignored downstream.

**Skipping practice activities is not free.** `os6_inf_prog` drops 18 of 50
activities and lectures 3, 6, 10 and 13 are a single practice activity each, so
they produce nothing at all. Use `--include-practice` to get them back.

### 2) gen-answers

**Step 2** — Answer questions using a locally-served model (vLLM).

```bash
astft gen-answers --help
```

Options:

- `--questions FILE` (default: `questions_derived.jsonl`) — run once per context variant
- `--output, -o FILE` (default: `answers.jsonl`)
- `--base-url TEXT` (default: `http://localhost:8000/v1`)
- `--model, -m TEXT` (default: `Qwen/Qwen3-14B`)
- `--api-key TEXT` (default: `EMPTY`)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft gen-answers --questions finetunedata/questions/questions_derived.jsonl --model Qwen/Qwen3-14B
```

Output format (`answers.jsonl` — one line per question):

```json
{"source": "uvod/programski_jezici.md", "system_message": "...", "question": "What is the difference...", "answer": "A compiler translates..."}
```

### 3) judge

**Step 3** — Score every answer with the teacher model, then have the teacher
improve it. This is where the teacher money goes, and it **stops at a judged
file**: no training data is written here. `make-sft` and `make-dpo` do that, for
free, as often as you like.

Two passes, two files. Pass 1 scores substance and Serbian separately into
`--scores-output`. Pass 2 writes the teacher's answer beside the model's into
`--written-output`, one row per question — question, the model's answer, the
teacher's answer, both scores and the mode.

`--write` decides how the teacher answers:

| mode | what the teacher does |
|---|---|
| `repair` | fixes the answer in place: deletes and corrects, keeps the author's own words |
| `rewrite` | answers afresh from the material, never seeing the model's attempt |
| `split` | rewrites the worst `--rewrite-share` by substance, repairs the rest |

The mode says nothing about SFT or DPO — either training file can be built from
either kind of row. What it changes is how far the teacher's answer travels from
the model's, and that distance is what a DPO pair is made of.

```bash
astft judge --help
```

Options:

- `--answers FILE` (default: `answers.jsonl`) — from step 2
- `--scores-output FILE` (default: `scores.jsonl`) — pass 1
- `--written-output FILE` (default: `written.jsonl`) — pass 2, the judged file
- `--model, -m TEXT` (default: `gpt-5.6-sol`)
- `--write [repair|rewrite|split]` (default: `split`)
- `--rewrite-share FLOAT` (default: `0.25`) — only with `--write split`
- `--concurrency INT` (default: `4`)
- `--resume / --no-resume` (default: resume)
- `--scores-only` — run pass 1 and stop
- `--limit INT` — first N answers only, for trying a change out

Example:

```bash
astft judge --answers finetunedata/answers/answers_full.jsonl \n    --scores-output finetunedata/td/scores_full.jsonl \n    --written-output finetunedata/td/written_full.jsonl \n    --write split --rewrite-share 0.25 --concurrency 4
```

Output format (`written.jsonl` — one line per question):

```json
{"source": "1_lekcija/1_1_cas.md", "system_message": "...", "question": "...",
 "answer": "what the local model said", "improved_answer": "what the teacher said",
 "mode": "repair", "substance_score": 82, "language_score": 91}
```

Both files are checkpointed every 250 records and `--resume` picks up where an
interrupted run left off. Pass 1 is short and cheap; pass 2 runs for hours, which
is why they are separate files — a pass-2 checkpoint cannot disturb the scores.

The judged file is the **expensive artefact**. Keep it. Everything below is
derived from it in under a minute, so a training file can be thrown away and
rebuilt in a different shape whenever a run needs one.

### 4) make-sft

Build an SFT file from a judged file. The teacher's answer is the target; the
model's is ignored. No API calls.

```bash
astft make-sft --help
```

Options:

- `--written FILE` (default: `written.jsonl`)
- `--output, -o FILE` (default: `training_data.jsonl`)
- `--rows [all|repair|rewrite]` (default: `all`)
- `--shape [prompt-completion|messages]` (default: `prompt-completion`)
- `--max-tokens INT` — drop rows longer than this once rendered
- `--tokenizer-model TEXT` (default: `Qwen/Qwen3-14B`) — whose template measures length

**`--shape` is not cosmetic.** `prompt-completion` makes TRL resolve
`completion_only_loss` to True, so the loss covers the answer tokens only.
`messages` is a language-modelling dataset: the loss covers the whole sequence,
lesson material included. The material is handed to the model at inference and
never generated, and it is roughly three quarters of the tokens — so `messages`
spends most of the gradient on a task the model never performs. Measured on
belebele-sr against the base model: `t7-sft` (whole-sequence loss, April data)
lost 4.4 points, `derived-sft-02` (`prompt-completion`, September data) lost
0.8. The data differs too, so this is suggestive rather than controlled;
`derived-sft-01`, the September `messages` run, was never evaluated.

**`--max-tokens` drops rows, it does not truncate them.** Truncation removes the
end of the sequence, which is the answer, and trains the model to stop mid
sentence. Length is measured through the model's own chat template, because a
budget in characters cuts a different set of rows than `--max-seq-len` does.

#### The two styles we have trained

Same judged file, same teacher answers, no new API calls — only which tokens are
scored differs.

**Masked** — loss on the answer only. What `derived-sft-02` and `full-sft-01`
trained on, and the default. About **27%** of tokens enter next-token prediction:
the lesson context, the `<|im_start|>assistant` header and the empty
`<think></think>` block are all excluded.

```bash
astft make-sft --written finetunedata/td/written_full.jsonl \
    -o finetunedata/td/sft_full.jsonl \
    --rows all --shape prompt-completion --max-tokens 16384
```

**Unmasked** — loss over the whole sequence. What `derived-sft-01` and the older
`t7` runs trained on. **100%** of tokens enter next-token prediction, lesson
material included.

```bash
astft make-sft --written finetunedata/td/written_derived.jsonl \
    -o finetunedata/td/sft_derived.jsonl \
    --rows repair --shape messages
```

On belebele-sr, the masked `derived-sft-02` lost **0.8 pp** against the base
model and the unmasked April `t7-sft` lost **4.4 pp**. `derived-sft-01` was
never evaluated, so there is no same-data comparison yet.

### 5) make-dpo

Build DPO pairs from a judged file: `chosen` is the teacher's answer, `rejected`
is the model's. No API calls.

```bash
astft make-dpo --help
```

Options:

- `--written FILE` (default: `written.jsonl`)
- `--output, -o FILE` (default: `training_data_dpo.jsonl`)
- `--rows [all|repair|rewrite]` (default: `all`)
- `--max-tokens INT`, `--tokenizer-model TEXT` — as above, measured on the longer side

**Reach for `--rows repair` first.** A repair pair differs from its original by
deletions and corrections only: same words, same length, faults gone. The model
has to learn something to tell the sides apart. A rewrite pair is two unrelated
texts whose rejected side averages several times longer, and a run on rewrite
pairs alone reached zero loss in **40 of 219 steps** with reward accuracy 1.00 —
it learnt nothing for the remaining 82% of the run, and what it did learn was
partly "be brief".

Pairs whose sides are identical — the teacher changed nothing — are dropped:
they put the loss at log 2 with a zero gradient.

```bash
astft make-dpo --written finetunedata/td/written_full.jsonl \n    -o finetunedata/td/dpo_full.jsonl --rows repair --max-tokens 8192
```

Output format (one line per pair, conversational so TRL applies the chat
template):

```json
{"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "chosen": [{"role": "assistant", "content": "the teacher's answer"}],
 "rejected": [{"role": "assistant", "content": "the model's answer"}]}
```

### 6) sft-finetune

Run LoRA SFT fine-tuning on a CUDA GPU.

```bash
astft sft-finetune --help
```

Key options:

- `--data FILE` (default: `training_data.jsonl`)
- `--model, -m TEXT` (default: `Qwen/Qwen3-14B`)
- `--output-dir, -o DIRECTORY` (default: `./finetune-output`)
- `--epochs INT`
- `--batch-size INT`
- `--grad-accum INT`
- `--lr FLOAT`
- `--max-seq-len INT`
- `--lora-r INT`
- `--lora-alpha INT`
- `--lora-dropout FLOAT`
- `--use-4bit / --no-4bit` (default: no-4bit)

Example:

```bash
astft sft-finetune --data training_data.jsonl --model Qwen/Qwen3-14B --output-dir ./finetune-output
```

### 7) dpo-finetune

Run DPO alignment from an existing SFT adapter.

```bash
astft dpo-finetune --help
```

Required options:

- `--adapter DIRECTORY`
- `--training-data FILE`

Other options:

- `--model TEXT` (optional base model override)
- `--output-dir, -o DIRECTORY` (default: `./dpo-output`)
- `--epochs INT`
- `--batch-size INT`
- `--grad-accum INT`
- `--lr FLOAT`
- `--max-seq-len INT`
- `--beta FLOAT`
- `--use-4bit / --no-4bit` (default: no-4bit)

Example:

```bash
astft dpo-finetune --adapter ./finetune-output --training-data training_data_dpo.jsonl --output-dir ./dpo-output
```

## Typical Pipeline

Steps 1-3 cost API calls. Everything after them is free and repeatable.

### Generate (costs money, run once)

```bash
# Step 1 — questions, two passes. Pass 1 needs the local model on port 8000.
astft gen-questions --input-dir ./dataset/os6_inf_prog --output-dir finetunedata/questions/

# Step 2 — the local model answers. Once per context variant: that is the experiment.
astft gen-answers --questions finetunedata/questions/questions_derived.jsonl \n    --output finetunedata/answers/answers_derived.jsonl
astft gen-answers --questions finetunedata/questions/questions_full.jsonl \n    --output finetunedata/answers/answers_full.jsonl

# Step 3 — the teacher scores, then repairs or rewrites. Hours. Resumable.
astft judge --answers finetunedata/answers/answers_full.jsonl \n    --scores-output finetunedata/td/scores_full.jsonl \n    --written-output finetunedata/td/written_full.jsonl \n    --write split --rewrite-share 0.25
```

### Build a training file (free, as many times as you like)

```bash
# SFT on everything, answer-only loss, capped so no row is truncated
astft make-sft --written finetunedata/td/written_full.jsonl \n    -o finetunedata/td/sft_full.jsonl --rows all --max-tokens 8192

# DPO on the hard pairs only
astft make-dpo --written finetunedata/td/written_full.jsonl \n    -o finetunedata/td/dpo_full.jsonl --rows repair --max-tokens 8192
```

### Train — one or the other, not chained

```bash
astft sft-finetune --data finetunedata/td/sft_full.jsonl \n    --output-dir ./finetune-output/full-sft-01 \n    --max-seq-len 8192 --epochs 2 --batch-size 1 --grad-accum 16

astft dpo-finetune --adapter ./finetune-output/full-sft-01 \n    --training-data finetunedata/td/dpo_full.jsonl \n    --output-dir ./finetune-output/full-dpo-01 --max-seq-len 8192
```

**Do not chain SFT into DPO by reflex.** Running DPO on top of a freshly trained
SFT adapter, with `rejected` coming from the *pre-SFT* model, gives pairs the
adapter already separates at initialisation: the objective is satisfied within a
few dozen steps and the rest of the run is near-zero gradient. If you want DPO
after SFT, its rejected side should come from the SFT model itself (re-run
`gen-answers` against the adapter, then `judge` again), or at minimum from
`make-dpo --rows repair`.

Set `--max-seq-len` from the data, not from the default. The script defaults to
2048; measured against the real chat template that truncates 41.5% of the derived
set and 90% of the full set, and truncation takes the answer. `make-sft
--max-tokens` and `--max-seq-len` should carry the same number.

## Serving

`serve.sh` starts vLLM with LoRA adapter enabled.

```bash
bash serve.sh
```

Environment variables used by `serve.sh`:

- `BASE_MODEL` (default: `Qwen/Qwen3-14B`)
- `ADAPTER_PATH` (default: `./finetune-output`)
- `ADAPTER_NAME` (default: `Qwen3-14B-ft`)
- `MAX_LORA_RANK` (default: `16`)
- `PORT` (default: `8000`)
