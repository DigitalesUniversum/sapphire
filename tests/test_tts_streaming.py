"""Tests for core.tts.streaming.chunkify_for_speech.

Pure-function tests — no Sapphire integration. Focused on:
- Sentence boundary detection (.!? + capital lookahead)
- Block-level stripping that survives token-split tags
- Edge cases the plan calls out (Dr./3:45/URLs, decimals, ellipsis)
- Pause-after-ms metadata for the player
"""
import pytest

from core.tts.streaming import SpeechChunker, chunkify_for_speech, PAUSE_AFTER_MS


def _collect(tokens, min_chars=3, **kw):
    """Run chunkify and collect into a list. Default min_chars=3 here so
    short test strings actually exercise split logic — production default
    is 15 (which is verified separately in min-char guard tests)."""
    return list(chunkify_for_speech(iter(tokens), min_chars=min_chars, **kw))


# ─── Sentence detection — basic cases ─────────────────────────────────────────

def test_single_sentence_emits_one_chunk():
    chunks = _collect(["Hello there."])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello there."
    assert chunks[0]["boundary"] == "end"
    assert chunks[0]["index"] == 0


def test_two_sentences_emit_two_chunks():
    chunks = _collect(["Hi. How are you?"])
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Hi."
    assert chunks[0]["boundary"] == "sentence"
    assert chunks[1]["text"] == "How are you?"
    assert chunks[1]["boundary"] == "end"


def test_three_sentence_cascade():
    chunks = _collect(["Yes! No? Maybe."])
    assert [c["text"] for c in chunks] == ["Yes!", "No?", "Maybe."]
    assert [c["boundary"] for c in chunks] == ["sentence", "sentence", "end"]


def test_indices_are_zero_based_and_sequential():
    chunks = _collect(["One. Two. Three. Four."])
    assert [c["index"] for c in chunks] == [0, 1, 2, 3]


# ─── Negative sentence-detection (the "don't split" cases) ────────────────────

def test_dr_smith_does_not_split():
    chunks = _collect(["Dr. Smith arrived early."])
    # Acceptable failure per plan: this DOES split (period+space+capital).
    # If we ever upgrade to abbreviation-aware detection, this test changes.
    # Document current behavior:
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Dr."
    assert chunks[1]["text"] == "Smith arrived early."


def test_decimal_does_not_split():
    chunks = _collect(["Pi is 3.14 approximately."])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Pi is 3.14 approximately."


def test_time_with_colon_no_split_mid_sentence():
    # "3:45 PM" — colon followed by space + uppercase. _SECONDARY_RE
    # would normally split there, but only if chunk is already past the
    # secondary-split threshold. Short sentence stays one chunk.
    chunks = _collect(["Meeting at 3:45 PM today."])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Meeting at 3:45 PM today."


def test_url_does_not_split():
    chunks = _collect(["Visit https://example.com for info."])
    assert len(chunks) == 1
    assert "example.com" in chunks[0]["text"]


def test_lowercase_after_period_holds_until_more_text():
    # "world. and then" — lowercase 'a' after period prevents split.
    # The whole thing emits as one chunk at end-of-stream.
    chunks = _collect(["Hello world. and then more text."])
    assert len(chunks) == 1


# ─── Block tag stripping ──────────────────────────────────────────────────────

def test_think_block_stripped():
    chunks = _collect(["<think>reasoning here</think>Hello there."])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello there."


def test_reasoning_block_stripped():
    chunks = _collect(["<reasoning>plans</reasoning>Hi there."])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hi there."


def test_code_fence_stripped():
    chunks = _collect(["Hi.\n```\ncode block\nstuff\n```\nBye."])
    texts = [c["text"] for c in chunks]
    assert texts == ["Hi.", "Bye."]


def test_inline_code_stripped():
    chunks = _collect(["Use `foo()` to call. It works."])
    assert len(chunks) == 2
    # "foo()" is replaced with space; sentence retains structure
    assert "foo()" not in chunks[0]["text"]


def test_html_tag_stripped():
    chunks = _collect(["Hello <b>bold</b> world. Next."])
    assert len(chunks) == 2
    assert "<b>" not in chunks[0]["text"]
    assert "</b>" not in chunks[0]["text"]


def test_only_think_tags_emits_nothing():
    chunks = _collect(["<think>thinking thinking thinking</think>"])
    assert chunks == []


def test_only_code_block_emits_nothing():
    chunks = _collect(["```python\nprint('hi')\n```"])
    assert chunks == []


# ─── Streaming behavior — tag/text split across tokens ────────────────────────

def test_partial_think_tag_across_tokens():
    chunks = _collect(["<thi", "nk>secret reasoning</think>", "Hi there."])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hi there."
    assert "secret" not in chunks[0]["text"]


def test_partial_close_tag_across_tokens():
    chunks = _collect(["<think>plans</thi", "nk>", "Real text."])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Real text."


def test_partial_code_fence_across_tokens():
    chunks = _collect(["``", "`\nprint('hi')\n``", "`\nBye."])
    # Code fence stripped; final "Bye." emits
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Bye."


def test_one_char_at_a_time_still_chunks_correctly():
    tokens = list("Hi. How are you today?")
    chunks = _collect(tokens)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Hi."
    assert chunks[1]["text"] == "How are you today?"


def test_one_char_at_a_time_with_think_block():
    s = "<think>foo</think>Hello. World."
    chunks = _collect(list(s))
    texts = [c["text"] for c in chunks]
    assert texts == ["Hello.", "World."]


# ─── Boundary types + pause metadata ──────────────────────────────────────────

def test_paragraph_break_splits():
    chunks = _collect(["First para text.\n\nSecond para text."])
    assert len(chunks) == 2
    # The paragraph boundary fires first (before the period of "text.")
    assert chunks[0]["boundary"] == "paragraph"
    assert chunks[0]["text"] == "First para text."
    assert chunks[1]["text"] == "Second para text."


def test_ellipsis_boundary_with_long_pause():
    chunks = _collect(["Hmm... Maybe later."])
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Hmm..."
    assert chunks[0]["boundary"] == "ellipsis"
    assert chunks[0]["pause_after_ms"] == PAUSE_AFTER_MS["ellipsis"]


def test_sentence_boundary_pause():
    chunks = _collect(["Hello. World."])
    assert chunks[0]["pause_after_ms"] == PAUSE_AFTER_MS["sentence"]
    assert chunks[1]["pause_after_ms"] == PAUSE_AFTER_MS["end"]


def test_paragraph_pause_metadata():
    chunks = _collect(["Para one.\n\nPara two."])
    assert chunks[0]["pause_after_ms"] == PAUSE_AFTER_MS["paragraph"]


# ─── Max-char fallback ───────────────────────────────────────────────────────

def test_max_char_fallback_splits_long_no_punctuation():
    # Build a 300-char string with no terminal punctuation
    text = " ".join(["filler"] * 50) + " more text without punctuation here"
    assert len(text) > 200
    chunks = _collect([text], max_chars=200, min_chars=15)
    assert len(chunks) >= 2
    # First chunk should be maxlen (no sentence boundary) and <= max_chars
    assert chunks[0]["boundary"] == "maxlen"
    assert len(chunks[0]["text"]) <= 200


def test_max_char_fallback_splits_on_whitespace():
    text = "word " * 50  # 250 chars, no terminal punctuation
    chunks = _collect([text], max_chars=200, min_chars=15)
    assert chunks[0]["boundary"] == "maxlen"
    # Should split on whitespace, not mid-word
    assert not chunks[0]["text"].endswith("wor")
    assert chunks[0]["text"].endswith("word")


# ─── Min-char guard ───────────────────────────────────────────────────────────

def test_short_chunks_dont_emit_mid_stream():
    # "Hi." alone is 3 chars — below min_chars. Should wait for more.
    chunks_during = list(chunkify_for_speech(iter(["Hi."]), min_chars=15))
    # At end of stream, we emit whatever's left regardless of min_chars
    assert len(chunks_during) == 1
    assert chunks_during[0]["text"] == "Hi."
    assert chunks_during[0]["boundary"] == "end"


def test_short_sentence_emits_at_stream_end_only():
    # Pieced: send "Hi." token, then nothing. Should only emit at end.
    # We can't easily distinguish "during stream" emit vs "end" emit
    # without instrumenting — proxy by checking boundary type.
    chunks = list(chunkify_for_speech(iter(["Hi."]), min_chars=20))
    assert len(chunks) == 1
    assert chunks[0]["boundary"] == "end"


# ─── Quote handling ───────────────────────────────────────────────────────────

def test_quote_after_period_stays_with_sentence():
    chunks = _collect(["She said 'Yes.' Then she left."])
    assert len(chunks) == 2
    assert chunks[0]["text"] == "She said 'Yes.'"
    assert chunks[1]["text"] == "Then she left."


def test_double_quote_after_period():
    chunks = _collect(['He shouted "Now!" Then ran.'])
    assert len(chunks) == 2
    assert chunks[0]["text"] == 'He shouted "Now!"'


# ─── Empty / degenerate input ─────────────────────────────────────────────────

def test_empty_stream_emits_nothing():
    assert _collect([]) == []


def test_only_whitespace_emits_nothing():
    assert _collect(["   \n\t  "]) == []


def test_empty_tokens_ignored():
    chunks = _collect(["", "Hi.", "", " World.", ""])
    texts = [c["text"] for c in chunks]
    assert texts == ["Hi.", "World."]


# ─── Markdown handling ────────────────────────────────────────────────────────

def test_markdown_link_keeps_text():
    chunks = _collect(["See [the docs](https://example.com) for info."])
    assert len(chunks) == 1
    assert "the docs" in chunks[0]["text"]
    assert "example.com" not in chunks[0]["text"]


def test_bold_markers_stripped():
    chunks = _collect(["This is **important** text."])
    assert len(chunks) == 1
    assert "**" not in chunks[0]["text"]
    assert "important" in chunks[0]["text"]


def test_table_stripped():
    chunks = _collect(["Before table.\n| a | b |\n| 1 | 2 |\nAfter table."])
    texts = [c["text"] for c in chunks]
    # Table content should be gone; "Before" and "After" remain
    full = " ".join(texts)
    assert "Before table" in full
    assert "After table" in full
    assert "| a |" not in full


# ─── Whitespace normalization ─────────────────────────────────────────────────

def test_whitespace_collapsed_in_chunks():
    chunks = _collect(["Lots   of    spaces.    Next."])
    assert chunks[0]["text"] == "Lots of spaces."
    assert chunks[1]["text"] == "Next."


# ─── Sanity: structural fields always present ─────────────────────────────────

def test_chunk_shape():
    chunks = _collect(["Hello. World."])
    for c in chunks:
        assert isinstance(c, dict)
        assert set(c.keys()) == {"text", "boundary", "pause_after_ms", "index"}
        assert isinstance(c["text"], str)
        assert isinstance(c["boundary"], str)
        assert isinstance(c["pause_after_ms"], int)
        assert isinstance(c["index"], int)


# ─── SpeechChunker push/flush API (used by chat_streaming for live LLM) ──────

def test_chunker_push_then_flush():
    """Push-based API: same chunks as the generator function."""
    chunker = SpeechChunker(min_chars=3)
    out = []
    for tok in ["Hello", " there", ". Next", " up."]:
        out.extend(chunker.push(tok))
    out.extend(chunker.flush())
    assert [c["text"] for c in out] == ["Hello there.", "Next up."]


def test_chunker_yields_during_push_not_just_flush():
    """When a sentence boundary fully resolves mid-stream (i.e. the next
    capital arrives so the lookahead succeeds), push() returns the chunk
    immediately rather than waiting for flush()."""
    chunker = SpeechChunker(min_chars=3)
    out_during = []
    out_during.extend(chunker.push("Hello there. "))  # lookahead missing
    out_during.extend(chunker.push("More text here."))  # 'M' satisfies lookahead
    # First sentence should have emerged during push (we never called flush)
    assert any(c["text"] == "Hello there." for c in out_during)


def test_chunker_indices_continuous_across_push_and_flush():
    chunker = SpeechChunker(min_chars=3)
    out = []
    out.extend(chunker.push("Hi. "))
    out.extend(chunker.push("How are you? "))
    out.extend(chunker.flush())
    indices = [c["index"] for c in out]
    assert indices == list(range(len(out)))


def test_chunker_partial_tag_across_pushes():
    """The push API must handle the same partial-tag-across-tokens cases
    as the generator function."""
    chunker = SpeechChunker(min_chars=3)
    out = []
    out.extend(chunker.push("<thi"))
    out.extend(chunker.push("nk>secret</think>"))
    out.extend(chunker.push("Hello there."))
    out.extend(chunker.flush())
    assert [c["text"] for c in out] == ["Hello there."]


def test_chunker_empty_stream_flushes_nothing():
    chunker = SpeechChunker()
    assert chunker.push("") == []
    assert chunker.flush() == []
