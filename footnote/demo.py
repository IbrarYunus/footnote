"""Demo mode: a stand-in for the Claude client that replays scripted model turns, so the app runs with no API key.

Only the model's words are scripted. The searches it "asks for" really run, and every citation is resolved
against the passages those searches actually returned. A quote that was not retrieved gets no citation.
"""

import time
from contextlib import contextmanager
from types import SimpleNamespace as NS

MODEL_NAME = "demo (scripted replies)"

# Each script is a list of model turns. A turn is either searches (with a short note) or the final answer,
# written as (text, quote) segments. A quote is a phrase from the passage, or (first, last) to span sentences.
SCRIPTS = {
    "how much annual leave do i earn, and can i carry it over?": [
        {"note": "Two parts here: how leave builds up, and the carry-over limit. Searching each.",
         "searches": ["annual leave accrual rate hours per pay period", "annual leave carry over maximum use or lose"]},
        {"answer": [
            ("Annual leave builds up with your length of federal service", "Annual leave accrues based on how long"),
            (": fewer than 3 years earns 4 hours per pay period, 3 to 15 years earns 6, and 15 or more earns 8.", ("- Fewer than 3 years", "- 15+ years")),
            (" The rate goes up on the first full pay period after your work anniversary.", "Your leave increases on the first full pay period"),
            ("\n\nYou can carry over at most 240 hours of annual leave from one leave year to the next.", "there is a 240 hour limit"),
            (" Anything above that shows as \"Use or Lose\": hours you forfeit at the end of the leave year if you do not take them. It applies to annual leave only.", ("Use or Lose is the number", "Use or Lose only applies")),
        ]},
    ],
    "what is the dress code for the office christmas party?": [
        {"note": "Looking for a dress code first.", "searches": ["office dress code"]},
        {"note": "Nothing about events. Trying the party angle before giving up.", "searches": ["holiday party office celebration"]},
        {"answer": [
            ("The handbook does not cover a Christmas party, or a dress code for any office event, so I can't answer that from it.", None),
            (" The closest it gets is everyday office wear. In Chicago there is no official dress code and business casual is what you will mostly see.", "There is no dress code officially"),
            (" In New York people range from casual t-shirts to business casual, and suits are rare.", ("TTS folx in the office usually range", "Suits would not be out of place")),
            (" For a specific event, ask whoever is organising it.", None),
        ]},
    ],
    "how do i get reimbursed for a conference?": [
        {"note": "Checking the conference rules, then how expenses are claimed back.",
         "searches": ["conference reimbursement"]},
        {"note": "Found approvals and vouchers. Need the deadline and review time for the voucher itself.",
         "searches": ["submit travel voucher expenses after trip Concur"]},
        {"answer": [
            ("You claim it back by creating a voucher in Concur, which covers official travel and other approved expenses you paid for yourself.", "You may be reimbursed for your expenses"),
            (" Start from the Vouchers tab and choose New Voucher.", "You can get started by navigating"),
            (" File it within 5 business days of getting back.", "within 5 business days of getting back"),
            (" The travel team reviews vouchers in 3-5 business days.", "Your voucher will be reviewed in 3-5 business days"),
            ("\n\nGet the event approved before you go: a conference under $2,500 including travel needs ten days to two weeks' notice.", ("- Ten days to two weeks", "Conference events that cost less than $2500")),
            (" Flights or rail booked outside Concur are reimbursed at your own risk and need a written justification.", ("Any airfare or Amtrak tickets booked outside", "provide a justification")),
        ]},
    ],
}

EXAMPLES = [
    "How much annual leave do I earn, and can I carry it over?",
    "How do I get reimbursed for a conference?",
    "What is the dress code for the office Christmas party?",
]

FALLBACK = (
    "Demo mode only has scripted replies for the example questions, so I can't write an answer to this one. "
    "The passages on the right are what the real search found for it. Add an ANTHROPIC_API_KEY to ask anything."
)


def _search_results(messages: list[dict]) -> list[dict]:
    """Every search_result block sent so far, in the order the API would index them."""
    found = []
    for message in messages:
        if message["role"] != "user" or isinstance(message["content"], str):
            continue
        for part in message["content"]:
            if part.get("type") == "tool_result" and isinstance(part.get("content"), list):
                found.extend(block for block in part["content"] if block.get("type") == "search_result")
    return found


def _locate(results: list[dict], quote) -> NS | None:
    first, last = (quote, quote) if isinstance(quote, str) else quote
    for result_index, result in enumerate(results):
        texts = [block["text"] for block in result["content"]]
        start = next((i for i, text in enumerate(texts) if first in text), None)
        if start is None:
            continue
        end = next((i for i in range(start, len(texts)) if last in texts[i]), None)
        if end is None:
            continue
        return NS(
            type="search_result_location", search_result_index=result_index, source=result["source"],
            title=result["title"], start_block_index=start, end_block_index=end + 1,
            cited_text="".join(texts[start:end + 1]),
        )
    return None


def _words(text: str):
    piece = ""
    for char in text:
        piece += char
        if char == " ":
            yield piece
            piece = ""
    if piece:
        yield piece


class _Stream:
    def __init__(self, messages: list[dict]):
        question = messages[0]["content"].strip().lower()
        turn = sum(1 for message in messages if message["role"] == "assistant")
        script = SCRIPTS.get(question) or [{"note": "", "searches": [messages[0]["content"]]}, {"answer": [(FALLBACK, None)]}]
        self.turn = script[min(turn, len(script) - 1)]
        self.results = _search_results(messages)
        self.content: list[dict] = []

    def __iter__(self):
        time.sleep(0.5)
        if "searches" in self.turn:
            segments = [(self.turn["note"], None)] if self.turn["note"] else []
        else:
            segments = self.turn["answer"]
        for index, (text, quote) in enumerate(segments):
            citation = _locate(self.results, quote) if quote else None
            block = {"type": "text", "text": text}
            if citation:
                block["citations"] = [vars(citation)]
                yield NS(type="content_block_delta", index=index, delta=NS(type="citations_delta", citation=citation))
            for piece in _words(text):
                time.sleep(0.012)
                yield NS(type="content_block_delta", index=index, delta=NS(type="text_delta", text=piece))
            self.content.append(block)
        for number, query in enumerate(self.turn.get("searches", [])):
            self.content.append({"type": "tool_use", "id": f"demo_{len(self.results)}_{number}", "name": "search_handbook", "input": {"query": query}})

    def get_final_message(self):
        content = self.content
        return NS(
            content=[NS(**block) for block in content],
            stop_reason="tool_use" if "searches" in self.turn else "end_turn",
            usage=NS(input_tokens=0, output_tokens=0),
            model=MODEL_NAME,
            to_dict=lambda: {"content": content},
        )


class _Messages:
    @contextmanager
    def stream(self, *, messages, **_):
        yield _Stream(messages)


class DemoClient:
    messages = _Messages()
