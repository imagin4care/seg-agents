"""What NV-Reason-CT writes, sent where it belongs as it arrives: the
reasoning it writes before `</think>` apart from the answer that follows.

The model's chat template opens the reasoning in the prompt (`<think>\\n`), so
the text generated is the reasoning, `</think>`, then the answer; with
reasoning off, the template writes an empty `<think></think>` block itself
and the text is the answer. A model may still open its text with `<think>`:
the tag is dropped, and what it opens is reasoning (an empty block leaves
none). The text arrives in pieces cut anywhere, a tag across two of them:
only what could be the start of a tag is held back until the next piece says.

Standard library only, so that it is tested without the model.
"""

OPEN, CLOSE = "<think>", "</think>"


def could_start(text, tag):
    """How many characters at the end of `text` could be the start of `tag`
    (fewer than the tag: a whole tag is found, not held)."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


class Router:
    """Takes the model's text piece by piece (`feed`) and writes it with
    `reason` or `answer`, each without the blank lines it starts with; `close`
    writes what was held back. `in_reasoning` then says whether the text ended
    before the reasoning did."""

    def __init__(self, thinking, reason, answer):
        self.thinking = thinking
        self.writers = {"reasoning": reason, "answer": answer}
        self.started = dict.fromkeys(self.writers, False)
        # None until the start of the text says which part it is.
        self.part = None
        self.held = ""

    @property
    def in_reasoning(self):
        return self.part == "reasoning"

    def feed(self, piece):
        self.held += piece
        if self.part is None:
            start = self.held.lstrip()
            if len(start) < len(OPEN) and OPEN.startswith(start):
                return
            if start.startswith(OPEN):
                self.held, self.part = start[len(OPEN):], "reasoning"
            else:
                self.part = "reasoning" if self.thinking else "answer"
        if self.part == "reasoning":
            end = self.held.find(CLOSE)
            if end < 0:
                cut = len(self.held) - could_start(self.held, CLOSE)
                self._write("reasoning", self.held[:cut])
                self.held = self.held[cut:]
                return
            self._write("reasoning", self.held[:end])
            self.held, self.part = self.held[end + len(CLOSE):], "answer"
        self._write("answer", self.held)
        self.held = ""

    def close(self):
        if self.part is None:
            self.part = "reasoning" if self.thinking else "answer"
        self._write(self.part, self.held)
        self.held = ""

    def _write(self, part, text):
        if not self.started[part]:
            text = text.lstrip()
        if text:
            self.started[part] = True
            self.writers[part](text)
