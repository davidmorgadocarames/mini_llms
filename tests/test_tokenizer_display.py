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


def test_formats_wikitext_section_header_as_its_own_uppercase_line():
    text = "under the command of Fatah . = = = Siege of Cape Town = = = When a large fleet"
    assert clean_for_display(text) == (
        "under the command of Fatah .\n\nSIEGE OF CAPE TOWN\n\nWhen a large fleet"
    )


def test_formats_single_equals_level_header():
    text = "band . = Background = After leaving"
    assert clean_for_display(text) == "band .\n\nBACKGROUND\n\nAfter leaving"


def test_header_at_start_of_text():
    text = "= = Background = = After leaving the band"
    assert clean_for_display(text) == "\n\nBACKGROUND\n\nAfter leaving the band"


def test_eot_and_header_together():
    text = "the end . <|endoftext|> = = = Newspaper sources = = = The New York Times"
    assert clean_for_display(text) == (
        "the end .\n\nNEWSPAPER SOURCES\n\nThe New York Times"
    )
