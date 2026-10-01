"""Code checks on one reasoning trace and answer from the local model.  No LLM.

Run before any teacher call is paid for, so a pair that is broken in a way code
can see never reaches the judge.  Two kinds of finding:

    hard_faults   the pair is unusable: degenerate output (loops, special tokens,
                  CJK, reasoning leaked into the answer, a runaway list, an answer
                  that stops on a colon), the wrong script, over the length caps,
                  or a lesson plan whose phases don't add up to the stated time.
    script_words  words a word-level fix can repair: a Latin letter inside a
                  Cyrillic word, a Serbian Latin word in a Cyrillic answer, a
                  letter Serbian Cyrillic doesn't have (Russian, Bulgarian).  Sent
                  to the judge as hints and checked again after the fixes.

The thresholds are set loose enough that a normal answer never trips them; the
pilot's distribution is what they were checked against.  The rules for scripts
and plan minutes are run-02's cleaning rules.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

CYR = "а-яА-ЯђјљњћџЂЈЉЊЋЏ"
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
MIXED = re.compile(rf"\b(?=\w*[{CYR}])(?=\w*[a-zA-Z])\w+\b")
# A Latin name with a Cyrillic case ending ("Python-у" without the hyphen), and
# identifiers (underscore, CamelCase): not script errors.
NAME_END = re.compile(rf"^[A-Za-z]{{2,}}[{CYR}]{{1,3}}$")
IDENT = re.compile(r"_|[a-z][А-Я]|[а-я][A-Z]|[A-Z]{2}")
SERB_LATIN = re.compile(r"\b\w*[čćšžđČĆŠŽĐ]\w*\b")
# Cyrillic letters Serbian doesn't use: Russian, Bulgarian, Ukrainian.
NON_SERBIAN = re.compile(r"\b\w*[ёйщъыьэюяєіїґЁЙЩЪЫЬЭЮЯЄІЇҐ]\w*\b")
PICTO = re.compile("[\U0001F000-\U0001FAFF]️? ?")

CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯！-｠]")
SPECIAL = re.compile(r"<\|[a-z_]+\|>|</?think>|</?tool_call>")
BAD_CHAR = re.compile("[�\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# The model's own thinking written into the answer, in English.
META = re.compile(
    r"\b(?:the teacher (?:wants|asked|is asking)|I need to|I should|I'll|"
    r"let me|okay, so|first, I)\b|\b(?:wait|hmm),",
    re.I,
)
NUMBERED = re.compile(r"^\s*\d+[.)]\s", re.M)
FENCE = re.compile(r"^\s*```", re.M)
TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")
URL = re.compile(r"https?://\S+")
LETTER = re.compile(r"[^\W\d_]")
MIN = re.compile(r"(\d{1,3})\s*(?:–|-|до)?\s*(\d{1,3})?\s*мин")
# A request for a lesson plan, by its nouns ("припрему", "план часа", "ток
# часа"), not the verb "припреми", which asks for tests and homework too.
PLAN = re.compile(r"(?<!\w)(?:припрем[ауе]|план[ауо]?м?|ток часа|скиц[ауе] часа)(?!\w)", re.I)

ANSWER_CAP = 12_000       # characters; run-02's cap on targets
TRACE_CAP = 10_000        # characters; think-01's cap on traces
ANSWER_MIN = 80
TRACE_MIN = 100
LATIN_SHARE = 0.30        # of the letters in the answer's prose
MAX_NUMBERED = 60         # base Qwen3 without thinking once wrote 91
# Repetition.  Checked against the 32B's think-01 answers: a worksheet repeats
# its fill-in line, a test its "Одговор: ____", a list the same link, a trace
# re-quotes a formula three times.  A real loop repeats more, and usually runs
# into --max-tokens, where gen-answers drops it; these catch the ones that
# stopped on their own.
LINE_REPEATS = 4          # the same long line this often is a loop
LINE_MIN = 40             # characters, of which at least LINE_LETTERS letters
LINE_LETTERS = 15
ANSWER_NGRAM, ANSWER_NGRAM_REPEATS = 12, 4
TRACE_NGRAM, TRACE_NGRAM_REPEATS = 25, 5
COMPRESS_MIN = 0.15       # zlib ratio; normal answers sit well above
COMPRESS_FROM = 2_000     # only measured on answers this long


def prose(text: str) -> str:
    return CODE.sub(" ", text)


def latin_share(text: str) -> float:
    cyr = len(re.findall(f"[{CYR}]", text))
    lat = len(re.findall("[a-zA-Z]", text))
    return lat / (cyr + lat) if cyr + lat else 0.0


def ngram_repeats(text: str, n: int) -> int:
    """How often the most frequent word n-gram occurs, links aside."""
    words = re.findall(r"\w+", URL.sub(" ", text).lower())
    if len(words) < n:
        return 0
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(counts.values())


def line_repeats(text: str) -> int:
    """How often the most frequent long line occurs, table rules aside."""
    lines = [
        l.strip() for l in text.splitlines()
        if len(l.strip()) >= LINE_MIN and not TABLE_RULE.match(l.strip())
        and len(LETTER.findall(l)) >= LINE_LETTERS
    ]
    return max(Counter(lines).values(), default=0)


def compress_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(zlib.compress(raw, 9)) / len(raw) if raw else 1.0


def stated_minutes(question: str) -> int | None:
    q = question.lower()
    if "двочас" in q:
        return 90
    m = re.search(r"(\d{2,3})\s*(?:-?минут|мин\b)", q)
    return int(m.group(1)) if m else None


def _level(line: str) -> int | None:
    """Outline depth of a line that can carry a phase time; None if it can't."""
    s = line.lstrip()
    if s.startswith("|"):
        return None if re.match(r"^\|[\s:|-]+\|?$", s) else 0
    if s.startswith("#"):
        return len(s) - len(s.lstrip("#"))
    if s.startswith("**"):
        return 7
    if re.match(r"^(\d+[.)]|[-*])\s", s):
        return 8 + (len(line) - len(s))
    return None


def phase_minutes(answer: str) -> tuple[int | None, bool]:
    """Minutes summed over the shallowest outline level that holds >= 2 timed
    lines (the phases, not their sub-steps).  (sum, parsed_cleanly)."""
    by_level: dict[int, list[tuple[int, bool]]] = {}
    first = True
    for line in answer.splitlines():
        level = _level(line)
        if level is None:
            continue
        low = line.lower()
        m = MIN.search(line)
        if not m:
            continue
        # Header lines carry the total, not a phase: "Трајање: 45 минута",
        # "Укупно", or a title like "План часа (45 минута)" before any phase.
        if any(w in low for w in ("трајање", "укупн", "време:")):
            continue
        if re.match(r"^\s*[-*]?\s*\**\d+\s*минут\w*\.?\**\s*$", line):
            continue
        if first and any(w in low for w in ("план", "припрема за", "припрема часа", "час (")):
            first = False
            continue
        first = False
        by_level.setdefault(level, []).append((int(m.group(1)), bool(m.group(2))))
    for level in sorted(by_level):
        items = by_level[level]
        if len(items) >= 2:
            ranged = any(r for _, r in items)
            return sum(n for n, _ in items), not ranged
    return None, False


def minutes_fault(question: str, answer: str) -> bool:
    """Timed parts that don't fit the time the request states.

    A lesson plan's phases must add up to it exactly; anything else (a 10-minute
    demonstration, a 15-minute homework) must not run over it -- "највише 10
    минута" is met by 8.  Two versions of a plan in one answer ("верзија") may
    add up to twice it.
    """
    stated = stated_minutes(question)
    got, clean = phase_minutes(answer)
    if not (stated and got is not None and clean):
        return False
    if got == 2 * stated and answer.lower().count("верзиј") >= 2:
        return False
    return got != stated if PLAN.search(question) else got > stated


def hard_faults(question: str, answer: str, trace: str) -> list[str]:
    """Why this pair can't be used; empty if code sees nothing wrong."""
    return answer_faults(question, answer) + trace_faults(trace)


def trace_faults(trace: str) -> list[str]:
    if not trace.strip():
        return ["empty-trace"]
    why = []
    if len(trace) < TRACE_MIN:
        why.append("short-trace")
    if len(trace) > TRACE_CAP:
        why.append("trace-cap")
    if SPECIAL.search(trace):
        why.append("trace-special-token")
    if CJK.search(trace):
        why.append("trace-cjk")
    if BAD_CHAR.search(trace):
        why.append("trace-bad-char")
    if ngram_repeats(trace, TRACE_NGRAM) >= TRACE_NGRAM_REPEATS:
        why.append("trace-loop")
    return why


def answer_faults(question: str, answer: str) -> list[str]:
    if not answer.strip():
        return ["empty-answer"]
    why = []
    p = prose(answer)
    if len(answer) > ANSWER_CAP:
        why.append("answer-cap")
    if len(answer) < ANSWER_MIN:
        why.append("short-answer")
    if SPECIAL.search(answer):
        why.append("special-token")
    # Only outside code: a string literal may hold any script.
    if CJK.search(p):
        why.append("cjk")
    if BAD_CHAR.search(answer):
        why.append("bad-char")
    if META.search(p):
        why.append("meta")
    if latin_share(p) > LATIN_SHARE:
        why.append("latin")
    if len(NUMBERED.findall(answer)) > MAX_NUMBERED:
        why.append("runaway-list")
    if len(FENCE.findall(answer)) % 2:
        why.append("open-fence")
    # "Ево примера:" and nothing after it; not a ":::" block closer.
    last = [l for l in answer.splitlines() if l.strip()][-1].strip()
    if LETTER.search(last) and last.endswith(":") and not last.endswith("::"):
        why.append("ends-colon")
    if line_repeats(p) >= LINE_REPEATS:
        why.append("repeat-line")
    if ngram_repeats(p, ANSWER_NGRAM) >= ANSWER_NGRAM_REPEATS:
        why.append("loop")
    if len(answer) >= COMPRESS_FROM and compress_ratio(answer) < COMPRESS_MIN:
        why.append("compress")
    if minutes_fault(question, answer):
        why.append("minutes")
    return why


def script_words(answer: str) -> list[str]:
    """Words in the answer's prose that a word-level fix can repair."""
    p = prose(answer)
    words = {w for w in MIXED.findall(p) if not NAME_END.match(w) and not IDENT.search(w)}
    words |= set(NON_SERBIAN.findall(p))
    if latin_share(p) <= LATIN_SHARE:
        words |= set(SERB_LATIN.findall(p))
    return sorted(words)


_DIGRAPHS = [("dž", "џ"), ("Dž", "Џ"), ("DŽ", "Џ"), ("lj", "љ"), ("Lj", "Љ"), ("LJ", "Љ"),
             ("nj", "њ"), ("Nj", "Њ"), ("NJ", "Њ")]
_LETTERS = dict(zip("abcčćdđefghijklmnoprsštuvzžABCČĆDĐEFGHIJKLMNOPRSŠTUVZŽ",
                    "абцчћдђефгхијклмнопрсштувзжАБЦЧЋДЂЕФГХИЈКЛМНОПРСШТУВЗЖ"))


def _to_cyrillic(word: str) -> str:
    for latin, cyrillic in _DIGRAPHS:
        word = word.replace(latin, cyrillic)
    return "".join(_LETTERS.get(ch, ch) for ch in word).replace("й", "ј").replace("Й", "Ј")


def fix_script(answer: str) -> str:
    """The words `script_words` flags, written in Cyrillic, outside code only.

    The 14B switches script mid-word ("стresa", "контejnerа", "Заšто") and the
    Latin part is Serbian Latin, so transliterating it is exact; "й" is the
    Russian for "ј" ("слайдови").  Letters Serbian has no match for (q, w, x,
    y, and ю я ы ...) are left, so such a word stays flagged.
    """
    flagged = set(script_words(answer))
    if not flagged:
        return answer
    word = re.compile(r"\w+")
    out, last = [], 0
    for m in CODE.finditer(answer):
        out += [word.sub(lambda w: _to_cyrillic(w.group(0)) if w.group(0) in flagged else w.group(0),
                         answer[last : m.start()]), m.group(0)]
        last = m.end()
    out.append(word.sub(lambda w: _to_cyrillic(w.group(0)) if w.group(0) in flagged else w.group(0),
                        answer[last:]))
    return "".join(out)


def metrics(answer: str, trace: str) -> dict[str, float]:
    """The numbers the checks threshold, kept for the distribution report."""
    p = prose(answer)
    return {
        "answer_chars": len(answer),
        "trace_chars": len(trace),
        "latin_share": round(latin_share(p), 3),
        "numbered": len(NUMBERED.findall(answer)),
        "line_repeats": line_repeats(p),
        "ngram_repeats": ngram_repeats(p, ANSWER_NGRAM),
        "trace_ngram_repeats": ngram_repeats(trace, TRACE_NGRAM),
        "compress": round(compress_ratio(answer), 3),
    }
