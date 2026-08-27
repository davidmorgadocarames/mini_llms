from mini_llm.tokenizer.bpe import clean_for_display


def test_replaces_eot_token_with_paragraph_break():
    text = "type . \n <|endoftext|>  On 11 July 1947"
    assert clean_for_display(text) == "type .\n\nOn 11 July 1947"


def test_collapses_surrounding_whitespace():
    text = "end<|endoftext|>start"
    assert clean_for_display(text) == "end\n\nstart"


def test_handles_multiple_occurrences():
    text = "a <|endoftext|> b <|endoftext|> c"
    assert clean_for_display(text) == "a\n\nb\n\nc"


def test_leaves_text_without_the_token_unchanged():
    text = "just some ordinary generated prose"
    assert clean_for_display(text) == text
