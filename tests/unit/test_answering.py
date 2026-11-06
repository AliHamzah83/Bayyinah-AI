from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from src.answering import ARABIC_SYSTEM_PROMPT, AnswerGenerationService


class FixedChatCompletions:
    def __init__(self, answer: str):
        self.answer = answer
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FixedClient:
    def __init__(self, answer: str):
        self.chat = SimpleNamespace(completions=FixedChatCompletions(answer))


class FixedRetrieval:
    def __init__(self, hits):
        self.hits = hits

    def search(self, _query, *, top_k=5):
        return self.hits[:top_k]


def make_hit(text: str, *, score: float, filename: str = "validation.pdf", page: int = 1):
    return SimpleNamespace(
        document_id=f"{filename}-{page}",
        source_filename=filename,
        page_number=page,
        text=text,
        score=score,
    )


@pytest.mark.parametrize(
    ("query", "generated", "expected_answer"),
    [
        (
            "لخص الوثيقة في نقاط",
            "\n".join(f"- النقطة {number}" for number in range(1, 9)),
            "\n".join(f"- النقطة {number}" for number in range(1, 8)),
        ),
        (
            "ما الهدف من الوثيقة؟",
            "تهدف الوثيقة إلى توحيد الإجراءات.\nالدليل: ورد ذلك في النص [1].",
            "تهدف الوثيقة إلى توحيد الإجراءات.\nالدليل: ورد ذلك في النص [1].",
        ),
        (
            "من هو مرسل الوثيقة؟",
            "مرسل الوثيقة هو إدارة التخطيط.\nالدليل: يذكر النص اسم الإدارة [1].",
            "مرسل الوثيقة هو إدارة التخطيط.\nالدليل: يذكر النص اسم الإدارة [1].",
        ),
    ],
)
def test_answer_format_and_requested_arabic_behaviors(query, generated, expected_answer):
    hit = make_hit("تحدد الوثيقة هدفها وتذكر الجهة المرسلة.", score=0.9)
    client = FixedClient(generated)
    service = AnswerGenerationService(
        retrieval_service=FixedRetrieval([hit]),
        model="test-model",
        client=client,
    )

    result = service.answer(query)

    assert result.answer == (
        f"### الإجابة\n{expected_answer}\n\n"
        "### المصادر\n- [1] validation.pdf, page 1"
    )
    assert result.sources[0].filename == "validation.pdf"
    assert result.sources[0].page_number == 1
    assert len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", result.answer.split("### المصادر")[0])) <= 7
    assert client.chat.completions.request["messages"][0]["content"] == ARABIC_SYSTEM_PROMPT


def test_context_deduplicates_similar_chunks_and_logs_character_length(caplog):
    caplog.set_level("INFO", logger="src.answering")
    shared = "توضح الوثيقة آلية عمل النظام وخطوات تنفيذه " * 5
    duplicate = shared.replace("آلية", "آليةً")
    distinct = "تتولى إدارة التخطيط متابعة تنفيذ الإجراءات المعتمدة."
    client = FixedClient("توضح الوثيقة آلية العمل [1].")
    service = AnswerGenerationService(
        retrieval_service=FixedRetrieval(
            [
                make_hit(shared, score=0.95),
                make_hit(duplicate, score=0.9, filename="duplicate.pdf"),
                make_hit(distinct, score=0.8, page=2),
            ]
        ),
        model="test-model",
        client=client,
    )

    result = service.answer("ما آلية العمل؟")

    prompt_context = client.chat.completions.request["messages"][1]["content"]
    expected_context = (
        f"[1] الملف: validation.pdf، الصفحة: 1\n{shared}\n\n"
        f"[2] الملف: validation.pdf، الصفحة: 2\n{distinct}"
    )
    assert prompt_context.endswith(expected_context)
    assert len(result.sources) == 2
    assert result.sources[1].page_number == 2
    assert f"context_chars={len(expected_context)}" in caplog.text
