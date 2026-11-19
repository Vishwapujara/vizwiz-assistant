"""The inference cascade (PROJECT_PLAN 6.5, D-059).

    photo + question
      -> quality: if an unusable label is found (e.g. unrecognizable) -> retake guidance
      -> answerability: if P(unanswerable) >= threshold -> decline (+ retake tips)
      -> answer

Prompts are the training prompts (``formatting.SYSTEM_PROMPTS``) and decoding matches
the offline evaluation (greedy, same max tokens), so served replies are the ones that
were measured. ``mode`` sets how the stages overlap; the decision rule is the same in
every mode.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass, field

from vizwiz.data.config import load_data_config
from vizwiz.data.formatting import SYSTEM_PROMPTS
from vizwiz.eval.metrics import ANSWERABLE, UNANSWERABLE, clean_answer, parse_answerability
from vizwiz.eval.metrics import parse_quality as _parse_quality
from vizwiz.serve import guidance
from vizwiz.serve.config import ServeConfig
from vizwiz.serve.model_client import ModelClient

MAX_NEW_TOKENS = {"answer": 16, "answerability": 4, "quality": 48}  # = run_eval
TOP_LOGPROBS = 20  # = run_eval
QUALITY_LABELS = load_data_config().quality_labels


def messages(stage: str, image_url: str, question: str) -> list[dict]:
    user: list[dict] = [{"type": "image_url", "image_url": {"url": image_url}}]
    if stage != "quality":
        user.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[stage]},
        {"role": "user", "content": user},
    ]


def _is_prefix_of(token: str, word: str, other: str) -> bool:
    t = token.strip().lower()
    return bool(t) and word.startswith(t) and not other.startswith(t)


def unanswerable_score(first_top_logprobs: dict[str, float], text: str) -> tuple[float, bool]:
    """P(unanswerable) from the first token's top logprobs, by token text.

    Offline eval (D-040) uses token ids; over the API only token strings are
    available, so a token counts for a class when it is a prefix of that class word
    (any casing) and not of the other. The served score is checked against the
    offline score on the same photos at Step 5.1. Falls back to the generated label
    (1.0 / 0.0) when neither class appears; the bool says which was used.
    """
    mass = {UNANSWERABLE: 0.0, ANSWERABLE: 0.0}
    for tok, lp in first_top_logprobs.items():
        if _is_prefix_of(tok, UNANSWERABLE, ANSWERABLE):
            mass[UNANSWERABLE] += math.exp(lp)
        elif _is_prefix_of(tok, ANSWERABLE, UNANSWERABLE):
            mass[ANSWERABLE] += math.exp(lp)
    total = mass[UNANSWERABLE] + mass[ANSWERABLE]
    if total > 0:
        return mass[UNANSWERABLE] / total, True
    return (1.0 if parse_answerability(text) == UNANSWERABLE else 0.0), False


@dataclass
class StageTrace:
    stage: str
    adapter: str
    output: object
    latency_ms: float
    attempts: int
    used: bool = True  # False: ran speculatively and was discarded


@dataclass
class AssistResult:
    status: str  # answered | retake | unanswerable
    message: str
    answer: str | None
    issues: list[str]
    unanswerable_score: float | None
    tips: list[str]
    trace: list[StageTrace] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Cascade:
    def __init__(self, client: ModelClient, cfg: ServeConfig):
        self.client, self.cfg = client, cfg
        # stage -> adapter name on the model server; the registry swaps entries live (D-061)
        self.adapters = dict(cfg.adapters)

    async def _quality(self, url: str) -> tuple[list[str], StageTrace]:
        adapter = self.adapters["quality"]
        r = await self.client.chat(adapter, messages("quality", url, ""), MAX_NEW_TOKENS["quality"])
        labels, valid = _parse_quality(r.text, QUALITY_LABELS)
        issues = [lbl for lbl in QUALITY_LABELS if lbl in labels]  # fixed order
        out = {"issues": issues, "valid_json": valid}
        return issues, StageTrace("quality", adapter, out, r.latency_ms, r.attempts)

    async def _answerability(self, url: str, q: str) -> tuple[float, StageTrace]:
        adapter = self.adapters["answerability"]
        r = await self.client.chat(
            adapter, messages("answerability", url, q), MAX_NEW_TOKENS["answerability"],
            top_logprobs=TOP_LOGPROBS,
        )  # fmt: skip
        score, from_logprobs = unanswerable_score(r.first_top_logprobs, r.text)
        out = {"unanswerable_score": round(score, 4), "from_logprobs": from_logprobs}
        return score, StageTrace("answerability", adapter, out, r.latency_ms, r.attempts)

    async def _answer(self, url: str, q: str) -> tuple[str, StageTrace]:
        adapter = self.adapters["answer"]
        r = await self.client.chat(adapter, messages("answer", url, q), MAX_NEW_TOKENS["answer"])
        answer = clean_answer(r.text)
        return answer, StageTrace("answer", adapter, {"answer": answer}, r.latency_ms, r.attempts)

    async def run(self, image_url: str, question: str) -> AssistResult:
        t0 = time.perf_counter()
        rules, mode = self.cfg.rules, self.cfg.mode
        trace: list[StageTrace] = []
        answer_task = None
        if mode == "speculative":
            answer_task = asyncio.create_task(self._answer(image_url, question))
        try:
            if mode == "sequential":
                issues, tq = await self._quality(image_url)
                trace.append(tq)
                if set(issues) & set(rules.unusable_labels):
                    return self._finish(self._retake(issues), trace, t0)
                score, ta = await self._answerability(image_url, question)
                trace.append(ta)
            else:
                (issues, tq), (score, ta) = await asyncio.gather(
                    self._quality(image_url), self._answerability(image_url, question)
                )
                trace += [tq, ta]
                if set(issues) & set(rules.unusable_labels):
                    ta.used = False  # the photo is unusable whatever the question
                    return self._finish(self._retake(issues), trace, t0, answer_task)
            if score >= rules.answerability_threshold:
                return self._finish(self._decline(issues, score), trace, t0, answer_task)
            if answer_task is not None:
                answer, tans = await answer_task
                answer_task = None
            else:
                answer, tans = await self._answer(image_url, question)
            trace.append(tans)
            if not answer:  # an empty reply is treated as "cannot answer"
                return self._finish(self._decline(issues, score), trace, t0)
            tips = guidance.tips_for(issues, rules.max_tips)
            result = AssistResult("answered", answer, answer, issues, score, tips)
            return self._finish(result, trace, t0)
        finally:
            if answer_task is not None and not answer_task.done():
                answer_task.cancel()  # vLLM aborts the request when the client goes away

    def _retake(self, issues: list[str]) -> AssistResult:
        msg, tips = guidance.retake_message(issues, self.cfg.rules.max_tips)
        return AssistResult("retake", msg, None, issues, None, tips)

    def _decline(self, issues: list[str], score: float) -> AssistResult:
        msg, tips = guidance.decline_message(issues, self.cfg.rules.max_tips)
        return AssistResult("unanswerable", msg, None, issues, score, tips)

    def _finish(self, result, trace, t0, answer_task=None) -> AssistResult:
        if answer_task is not None and answer_task.done() and not answer_task.cancelled():
            if answer_task.exception() is None:
                _, tans = answer_task.result()
                tans.used = False
                trace.append(tans)
        result.trace = trace
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        ta = next((t for t in trace if t.stage == "answerability"), None)
        result.unanswerable_score = ta.output["unanswerable_score"] if ta else None
        return result
