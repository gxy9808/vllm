# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping, Sequence
from contextlib import suppress
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, overload

if TYPE_CHECKING:
    from transformers import BatchEncoding

    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


_VOCAB_UPPER_BOUND_CACHE_KEY = "_vllm_vocab_upper_bound_cache"


def get_tokenizer_vocab_upper_bound(tokenizer: "TokenizerLike") -> int:
    """Return the exclusive upper bound of token IDs exposed by a tokenizer.

    Tokenizer implementations do not agree on which vocabulary metadata
    includes added/special tokens:

    * ``get_vocab()`` may expose a sparse map and/or omit added tokens;
    * ``len(tokenizer)`` is generally a count, and therefore underestimates a
      sparse vocabulary;
    * ``vocab_size`` usually describes only the base vocabulary; and
    * older/custom ``TokenizerLike`` implementations may only expose
      ``max_token_id``.

    Use the maximum of every available source so callers get one conservative
    *exclusive* bound.  ``max_token_id`` is the legacy inclusive property and
    is converted to the same representation here.
    """
    cached = getattr(tokenizer, _VOCAB_UPPER_BOUND_CACHE_KEY, None)
    if isinstance(cached, int):
        return cached

    upper_bounds: list[int] = []

    def add_upper_bound(value: Any, source: str) -> None:
        """Validate and add an exclusive vocabulary bound."""
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"Tokenizer {source} must be a non-negative integer, got {value!r}."
            )
        value = int(value)
        if value < 0:
            raise ValueError(
                f"Tokenizer {source} must be a non-negative integer, got {value}."
            )
        upper_bounds.append(value)

    def add_token_ids(values: Any, source: str) -> None:
        """Add an inclusive token-ID collection as an exclusive bound."""
        if values is None:
            return
        try:
            iterator = iter(values)
        except TypeError as exc:
            raise ValueError(
                f"Tokenizer {source} must be an iterable of token IDs."
            ) from exc
        for token_id in iterator:
            if token_id is None:
                # A few Hugging Face tokenizers use ``None`` for an unset
                # special token.  It does not contribute to the bound.
                continue
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise ValueError(
                    f"Tokenizer {source} must contain non-negative integer "
                    f"token IDs, got {token_id!r}."
                )
            token_id = int(token_id)
            if token_id < 0:
                raise ValueError(
                    f"Tokenizer {source} must contain non-negative integer "
                    f"token IDs, got {token_id}."
                )
            upper_bounds.append(token_id + 1)

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        # TokenizerLike exposes get_vocab(), but third-party tokenizer
        # implementations may intentionally leave it unsupported.
        try:
            tokenizer_vocab = get_vocab()
        except NotImplementedError:
            tokenizer_vocab = None
        if tokenizer_vocab is not None:
            if not isinstance(tokenizer_vocab, Mapping):
                raise ValueError(
                    "Tokenizer get_vocab() must return a mapping of token "
                    "strings to integer token IDs."
                )
            add_token_ids(tokenizer_vocab.values(), "get_vocab()")

    # ``get_added_vocab`` is separate from ``get_vocab`` for some custom HF
    # tokenizers.  Include it explicitly so high-ID special tokens cannot be
    # silently excluded just because the base map is non-empty.
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        try:
            added_vocab = get_added_vocab()
        except NotImplementedError:
            added_vocab = None
        if added_vocab is not None:
            if not isinstance(added_vocab, Mapping):
                raise ValueError(
                    "Tokenizer get_added_vocab() must return a mapping of "
                    "token strings to integer token IDs."
                )
            add_token_ids(added_vocab.values(), "get_added_vocab()")

    # ``all_special_ids`` may contain IDs not present in either map.  Ignore
    # ``None`` entries (unset optional special tokens), but validate anything
    # else just like the vocabulary maps.
    try:
        all_special_ids = tokenizer.all_special_ids
    except (AttributeError, NotImplementedError):
        all_special_ids = None
    add_token_ids(all_special_ids, "all_special_ids")

    # ``len(tokenizer)`` counts entries and is not sufficient for sparse
    # vocabularies, but remains an important lower bound for contiguous
    # tokenizers and for tokenizers whose maps omit special tokens.
    try:
        tokenizer_len = len(tokenizer)
    except (TypeError, NotImplementedError):
        tokenizer_len = None
    add_upper_bound(tokenizer_len, "len(tokenizer)")

    try:
        tokenizer_vocab_size = tokenizer.vocab_size
    except (AttributeError, NotImplementedError):
        tokenizer_vocab_size = None
    add_upper_bound(tokenizer_vocab_size, "vocab_size")

    # Keep compatibility with older/custom tokenizers that implement only the
    # legacy inclusive ``max_token_id`` property.
    try:
        max_token_id = tokenizer.max_token_id
    except (AttributeError, NotImplementedError):
        max_token_id = None
    if max_token_id is not None:
        if isinstance(max_token_id, bool) or not isinstance(max_token_id, Integral):
            raise ValueError(
                f"Tokenizer max_token_id must be an integer, got {max_token_id!r}."
            )
        max_token_id = int(max_token_id)
        if max_token_id < -1:
            raise ValueError(
                f"Tokenizer max_token_id must be >= -1, got {max_token_id}."
            )
        upper_bounds.append(max_token_id + 1)

    if not upper_bounds:
        raise ValueError(
            "Tokenizer does not expose a usable vocabulary size or token-ID mapping."
        )

    upper_bound = max(upper_bounds)

    with suppress(AttributeError, TypeError):
        setattr(tokenizer, _VOCAB_UPPER_BOUND_CACHE_KEY, upper_bound)
    return upper_bound


class TokenizerLike(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TokenizerLike":
        raise NotImplementedError

    def num_special_tokens_to_add(self) -> int:
        raise NotImplementedError

    @property
    def all_special_tokens(self) -> list[str]:
        raise NotImplementedError

    @property
    def all_special_ids(self) -> list[int]:
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> int:
        raise NotImplementedError

    @property
    def is_fast(self) -> bool:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def max_token_id(self) -> int:
        """Inclusive largest token ID (``-1`` for an empty vocabulary)."""
        raise NotImplementedError

    @property
    def max_chars_per_token(self) -> int:
        raise NotImplementedError

    @property
    def truncation_side(self) -> str:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> "BatchEncoding":
        raise NotImplementedError

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError

    def get_added_vocab(self) -> dict[str, int]:
        raise NotImplementedError

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        raise NotImplementedError

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        raise NotImplementedError

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        raise NotImplementedError
