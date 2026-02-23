import pytest

# Adjust these imports based on your actual project structure
from optimus_dl.modules.tokenizer.implementations.inline_tokens import (
    InlineTokensTokenizer,
    InlineTokensTokenizerConfig,
    UnkStrategy,
)


@pytest.fixture
def base_tokens():
    # Including overlapping tokens to test greedy longest-match matching
    return ["A", "B", "C", "AB", "BC", "word", "!"]


def test_tokenizer_initialization_no_unk(base_tokens):
    """Test initialization and properties when UNK strategy does not create an UNK token."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.RAISE
    )
    tokenizer = InlineTokensTokenizer(config)

    assert tokenizer.vocab_size == len(base_tokens) + 2  # tokens + BOS + EOS
    assert tokenizer.bos_token_id == 7
    assert tokenizer.eos_token_id == 8
    assert getattr(tokenizer, "_unk_token_id", None) is None


def test_tokenizer_initialization_with_unk(base_tokens):
    """Test initialization and properties when UNK strategy creates an UNK token."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.UNK
    )
    tokenizer = InlineTokensTokenizer(config)

    assert tokenizer.vocab_size == len(base_tokens) + 3  # tokens + BOS + EOS + UNK
    assert tokenizer.bos_token_id == 7
    assert tokenizer.eos_token_id == 8
    assert tokenizer._unk_token_id == 9

    assert tokenizer.encode("$") == [7, 9, 8]
    assert tokenizer.encode("$A$") == [7, 9, 0, 9, 8]


def test_encode_greedy_matching(base_tokens):
    """Test that the tokenizer matches the longest possible tokens first."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.RAISE
    )
    tokenizer = InlineTokensTokenizer(config)

    # "ABC" could be "A", "B", "C" or "AB", "C" or "A", "BC".
    # Because of length sorting, "AB" (len 2) should match first, leaving "C" (len 1).
    # "AB" -> index 3, "C" -> index 2
    token_ids = tokenizer.encode("ABC")

    assert token_ids == [
        tokenizer.bos_token_id,
        3,  # "AB"
        2,  # "C"
        tokenizer.eos_token_id,
    ]

    # Test distinct word and symbol
    token_ids_2 = tokenizer.encode("word!")
    assert token_ids_2 == [
        tokenizer.bos_token_id,
        5,  # "word"
        6,  # "!"
        tokenizer.eos_token_id,
    ]


def test_encode_unk_strategy_raise(base_tokens):
    """Test that an unknown token raises a ValueError."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.RAISE
    )
    tokenizer = InlineTokensTokenizer(config)

    with pytest.raises(ValueError, match="Unknown token/characters encountered"):
        tokenizer.encode("ABXYZC")  # 'XYZ' is not in vocab


def test_encode_unk_strategy_ignore(base_tokens):
    """Test that an unknown token is silently ignored."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.IGNORE
    )
    tokenizer = InlineTokensTokenizer(config)

    # 'XYZ' should be skipped. "AB" and "C" should remain.
    token_ids = tokenizer.encode("ABXYZC")

    assert token_ids == [
        tokenizer.bos_token_id,
        3,  # "AB"
        2,  # "C"
        tokenizer.eos_token_id,
    ]


def test_encode_unk_strategy_unk(base_tokens):
    """Test that an unknown token is replaced by the UNK token ID."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.UNK
    )
    tokenizer = InlineTokensTokenizer(config)

    # 'XYZ' should become the UNK token ID
    token_ids = tokenizer.encode("ABXYZC")

    assert token_ids == [
        tokenizer.bos_token_id,
        3,  # "AB"
        tokenizer._unk_token_id,  # "XYZ"
        2,  # "C"
        tokenizer.eos_token_id,
    ]


def test_decode_standard(base_tokens):
    """Test decoding IDs back to a string, ignoring BOS/EOS."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.RAISE
    )
    tokenizer = InlineTokensTokenizer(config)

    # Decode: BOS, "word", "!", EOS
    ids_to_decode = [tokenizer.bos_token_id, 5, 6, tokenizer.eos_token_id]
    decoded_text = tokenizer.decode(ids_to_decode)

    assert decoded_text == "word!"


def test_decode_with_unk(base_tokens):
    """Test decoding handles the UNK token ID properly."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.UNK
    )
    tokenizer = InlineTokensTokenizer(config)

    # Decode: BOS, "AB", UNK, "C", EOS
    ids_to_decode = [
        tokenizer.bos_token_id,
        3,
        tokenizer._unk_token_id,
        2,
        tokenizer.eos_token_id,
    ]
    decoded_text = tokenizer.decode(ids_to_decode)

    assert decoded_text == "AB<UNK>C"


def test_decode_invalid_id(base_tokens):
    """Test that decoding an out-of-bounds ID raises an error."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.RAISE
    )
    tokenizer = InlineTokensTokenizer(config)

    # ID 99 doesn't exist
    with pytest.raises(ValueError, match="Invalid token ID: 99"):
        tokenizer.decode([tokenizer.bos_token_id, 99, tokenizer.eos_token_id])


def test_end_to_end_consistency(base_tokens):
    """Test that encoding and then decoding returns the original string."""
    config = InlineTokensTokenizerConfig(
        tokens=base_tokens, unk_strategy=UnkStrategy.RAISE
    )
    tokenizer = InlineTokensTokenizer(config)

    original_text = "word!ABC"
    encoded_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(encoded_ids)

    assert decoded_text == original_text
