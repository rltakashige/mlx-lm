import base64
import importlib
import json
import warnings
from functools import partial
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError()

    def add_token(self, token):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    @property
    def last_segment(self):
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self):
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
            if self._current_text.endswith("\ufffd") or (
                self._tokenizer.clean_up_tokenization_spaces
                and len(self._current_text) > 0
                and self._current_text[-1] == " "
            ):
                self._current_text = self._current_text[:-1]
        if self._current_text and self._current_text[-1] == "\n":
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space
        self._sep = "\u2581".encode()

        # Extract the tokens in a list from id to text
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            if value.startswith("<0x"):
                # Replace bytes with their value
                self.tokenmap[tokenid] = bytes([int(value[3:5], 16)])
            else:
                self.tokenmap[tokenid] = value.encode()

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = b""
        self.text = ""
        self.tokens = []

    def _try_flush(self, force=False):
        text = self._unflushed.replace(self._sep, b" ").decode("utf-8", "replace")
        if not force and text.endswith("\ufffd"):
            return
        if not self.text and self.trim_space and text and text[0] == " ":
            text = text[1:]
        self.text += text
        self._unflushed = b""

    def add_token(self, token):
        self.tokens.append(token)
        v = self.tokenmap[token]
        self._unflushed += v
        self._try_flush()

    def finalize(self):
        self._try_flush(force=True)
        self._unflushed = b""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    _byte_decoder = None
    _space_matches = (".", "?", "!", ",", "n't", "'m", "'s", "'ve", "'re")

    def __init__(self, tokenizer, trim_initial_space=True):
        self.clean_spaces = tokenizer.clean_up_tokenization_spaces
        self.trim_initial_space = trim_initial_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def _decode_bytes(self, seq):
        barr = bytearray()
        for c in seq:
            res = self._byte_decoder.get(c)
            if res is not None:
                barr.append(res)
            else:
                barr.extend(bytes(c, "utf-8"))
        return barr.decode("utf-8", "replace")

    def _maybe_trim_space(self, current_text):
        if len(current_text) == 0:
            return current_text
        elif current_text[0] != " ":
            return current_text
        elif not self.text and self.trim_initial_space:
            return current_text[1:]
        elif self.clean_spaces and current_text[1:].startswith(self._space_matches):
            return current_text[1:]
        return current_text

    def add_token(self, token):
        self.tokens.append(token)
        v = self.tokenmap[token] if token < len(self.tokenmap) else "!"
        self._unflushed += v
        text = self._decode_bytes(self._unflushed)

        # For multi-byte utf-8 wait until they are complete
        # For single spaces wait until the next token to clean it if needed
        if not text.endswith("\ufffd") and not (
            len(v) == 1 and self._byte_decoder.get(v[0]) == 32
        ):
            self.text += self._maybe_trim_space(text)
            self._unflushed = ""

    def finalize(self):
        current_text = self._decode_bytes(self._unflushed)
        self.text += self._maybe_trim_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls._byte_decoder is not None:
            return

        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


def _infer_thinking(tokenizer, thinking_markers=None):
    if (
        thinking_markers is None
        and tokenizer.__class__.__name__ == "_KimiK3TokenizerFast"
    ):
        thinking_markers = (
            "<|open|>think<|sep|>",
            "<|close|>think<|sep|>",
        )

    if thinking_markers is not None:
        think_start, think_end = thinking_markers
        return (
            think_start,
            think_end,
            tuple(tokenizer.encode(think_start, add_special_tokens=False)),
            tuple(tokenizer.encode(think_end, add_special_tokens=False)),
        )

    vocab = tokenizer.get_vocab()
    THINK_TOKENS = [
        ("<think>", "</think>"),
        ("<longcat_think>", "</longcat_think>"),
    ]

    # Single token thinking modes
    for think_start, think_end in THINK_TOKENS:
        if think_start in vocab and think_end in vocab:
            return (
                think_start,
                think_end,
                (vocab[think_start],),
                (vocab[think_end],),
            )

    # Multi token thinking modes
    if "<|channel>" in vocab and "<channel|>" in vocab:
        think_start = "<|channel>thought"
        think_end = "<channel|>"
        return (
            think_start,
            think_end,
            tuple(tokenizer.encode(think_start, add_special_tokens=False)),
            tuple(tokenizer.encode(think_end, add_special_tokens=False)),
        )

    return (None, None, None, None)


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(
        self,
        tokenizer,
        detokenizer_class=NaiveStreamingDetokenizer,
        eos_token_ids=None,
        chat_template=None,
        tool_call_start=None,
        tool_call_end=None,
        tool_parser=None,
        thinking_markers=None,
    ):
        self._tokenizer = tokenizer
        self._detokenizer_class = detokenizer_class
        self._eos_token_ids = (
            set(eos_token_ids)
            if eos_token_ids is not None
            else {tokenizer.eos_token_id}
        )
        (
            self._think_start,
            self._think_end,
            self._think_start_tokens,
            self._think_end_tokens,
        ) = _infer_thinking(tokenizer, thinking_markers)

        self._chat_template = chat_template
        self.has_chat_template = (
            tokenizer.chat_template is not None
            or chat_template is not None
            or bool(getattr(tokenizer, "has_chat_template", False))
        )
        self._tool_parser = tool_parser
        self._tool_call_start = tool_call_start
        self._tool_call_end = tool_call_end
        self._tool_call_start_tokens = None
        self._tool_call_end_tokens = None
        if tool_call_start is not None:
            self._tool_call_start_tokens = tuple(
                tokenizer.encode(tool_call_start, add_special_tokens=False)
            )
            self._tool_call_end_tokens = tuple(
                tokenizer.encode(tool_call_end, add_special_tokens=False)
            )

    def apply_chat_template(self, *args, tokenize=True, **kwargs):
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = self.has_thinking

        if self._chat_template is not None:
            out = self._chat_template(*args, **kwargs)
            if tokenize:
                out = self._tokenizer.encode(out, add_special_tokens=False)
            return out

        kwargs["return_dict"] = False
        return self._tokenizer.apply_chat_template(*args, tokenize=tokenize, **kwargs)

    def add_eos_token(self, token: str):
        token_id = None
        try:
            token_id = int(token)
        except ValueError:
            token_id = self._tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise ValueError(f"'{token}' is not a token for this tokenizer")

        self._eos_token_ids.add(token_id)

    def _find(self, tokens, sequence, start=None, end=None, reverse=False):
        start = start or 0
        end = end or len(tokens)
        outer_loop = (
            range(end - len(sequence), start - 1, -1)
            if reverse
            else range(start, end - len(sequence) + 1)
        )
        for i in outer_loop:
            if tokens[i] == sequence[0]:
                if all(tokens[i + j] == sequence[j] for j in range(1, len(sequence))):
                    return i
        return -1

    def find_think_start(self, tokens, start=None, end=None):
        return self._find(tokens, self._think_start_tokens, start=start, end=end)

    def rfind_think_start(self, tokens, start=None, end=None):
        return self._find(
            tokens, self._think_start_tokens, start=start, end=end, reverse=True
        )

    def find_think_end(self, tokens, start=None, end=None):
        return self._find(tokens, self._think_end_tokens, start=start, end=end)

    def rfind_think_end(self, tokens, start=None, end=None):
        return self._find(
            tokens, self._think_end_tokens, start=start, end=end, reverse=True
        )

    @property
    def has_thinking(self):
        return self._think_start is not None

    @property
    def think_start(self):
        return self._think_start

    @property
    def think_start_id(self):
        if self._think_start_tokens is None:
            return None
        if len(self._think_start_tokens) > 1:
            raise ValueError("The start thinking sequence is more than 1 token")
        return self._think_start_tokens[0]

    @property
    def think_start_tokens(self):
        return self._think_start_tokens

    @property
    def think_end(self):
        return self._think_end

    @property
    def think_end_id(self):
        if self._think_end_tokens is None:
            return None
        if len(self._think_end_tokens) > 1:
            raise ValueError("The end thinking sequence is more than 1 token")
        return self._think_end_tokens[0]

    @property
    def think_end_tokens(self):
        return self._think_end_tokens

    @property
    def has_tool_calling(self):
        return self._tool_call_start is not None

    @property
    def tool_call_start(self):
        return self._tool_call_start

    @property
    def tool_call_start_tokens(self):
        return self._tool_call_start_tokens

    @property
    def tool_call_end(self):
        return self._tool_call_end

    @property
    def tool_call_end_tokens(self):
        return self._tool_call_end_tokens

    @property
    def tool_parser(self):
        return self._tool_parser

    @property
    def detokenizer(self):
        """
        Get a stateful streaming detokenizer.
        """
        return self._detokenizer_class(self)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        elif attr == "eos_token_ids":
            return self._eos_token_ids
        elif attr.startswith("_"):
            return self.__getattribute__(attr)
        else:
            return getattr(self._tokenizer, attr)

    def __setattr__(self, attr, value):
        if attr in {"detokenizer", "eos_token_ids"}:
            if attr == "detokenizer":
                raise AttributeError("Cannot set the detokenizer.")
            elif attr == "eos_token_ids":
                self._eos_token_ids = set(value) if value is not None else set()
        elif attr.startswith("_"):
            super().__setattr__(attr, value)
        else:
            setattr(self._tokenizer, attr, value)


class NewlineTokenizer(PreTrainedTokenizerFast):
    """A tokenizer that replaces newlines with <n> and <n> with new line."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _preprocess_text(self, text):
        return text.replace("\n", "<n>")

    def _postprocess_text(self, text):
        return text.replace("<n>", "\n")

    def encode(self, text, **kwargs):
        return super().encode(self._preprocess_text(text), **kwargs)

    def encode_batch(self, texts, **kwargs):
        return super().encode_batch([self._preprocess_text(t) for t in texts], **kwargs)

    def decode(self, *args, **kwargs):
        return self._postprocess_text(super().decode(*args, **kwargs))

    def batch_decode(self, *args, **kwargs):
        decoded = super().batch_decode(*args, **kwargs)
        return [self._postprocess_text(d) for d in decoded]


AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)


_KIMI_K3_NUM_RESERVED_TOKENS = 256
_KIMI_K3_OPEN_TOKEN = "<|open|>"
_KIMI_K3_CLOSE_TOKEN = "<|close|>"
_KIMI_K3_SEPARATOR_TOKEN = "<|sep|>"
_KIMI_K3_END_OF_MESSAGE_TOKEN = "<|end_of_msg|>"
_KIMI_K3_THINKING_EFFORTS = {"low", "high", "max"}
_KIMI_K3_PATTERN = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


class _KimiK3TokenizerFast(PreTrainedTokenizerFast):
    has_chat_template = True

    @staticmethod
    def _open_tag(tag, attrs=()):
        attributes = "".join(
            f' {key}="{str(value).replace("&", "&amp;").replace(chr(34), "&quot;")}"'
            for key, value in attrs
        )
        return f"{_KIMI_K3_OPEN_TOKEN}{tag}{attributes}{_KIMI_K3_SEPARATOR_TOKEN}"

    @staticmethod
    def _close_tag(tag):
        return f"{_KIMI_K3_CLOSE_TOKEN}{tag}{_KIMI_K3_SEPARATOR_TOKEN}"

    def _validate_plain_text(self, text):
        for token_id in range(
            len(self) - _KIMI_K3_NUM_RESERVED_TOKENS,
            len(self),
        ):
            token = self.convert_ids_to_tokens(token_id)
            if token and token in text:
                raise ValueError(
                    "Kimi K3 chat text contains a reserved control token; "
                    "pass it as a raw prompt only if that is intentional"
                )

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        add_generation_prompt=True,
        tokenize=False,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_dict=False,
        **kwargs,
    ):
        """Render K3's text-only XTML conversation format without remote code.

        ``continue_final_message`` intentionally remains unsupported: K3's
        official renderer does not implement assistant prefilling semantics.
        """
        if tools:
            raise ValueError(
                "Kimi K3 tool rendering is unavailable in the safe local "
                "tokenizer fallback"
            )
        if kwargs.get("continue_final_message"):
            raise ValueError(
                "Kimi K3 continue_final_message is unavailable in the safe "
                "local tokenizer fallback"
            )

        is_batched = (
            isinstance(conversation, list)
            and bool(conversation)
            and isinstance(conversation[0], list)
        )
        conversations = conversation if is_batched else [conversation]
        reasoning_effort = kwargs.get("reasoning_effort")
        if "thinking" in kwargs:
            # Match K3's explicit ``thinking`` argument: None is falsy.
            thinking = bool(kwargs["thinking"])
        elif "enable_thinking" in kwargs and kwargs["enable_thinking"] is not None:
            thinking = bool(kwargs["enable_thinking"])
        else:
            # MLX/Exo use ``enable_thinking`` as a cross-model alias. Keep an
            # omitted/None alias equivalent to K3's default ``thinking=True``,
            # except for the standard ``reasoning_effort="none"`` shorthand.
            thinking = reasoning_effort != "none"
        # Match K3's serving renderer: the native ``thinking_effort`` kwarg
        # takes precedence over the standard ``reasoning_effort`` alias.
        thinking_effort = kwargs.get(
            "thinking_effort",
            "max" if reasoning_effort is None else reasoning_effort,
        )
        if thinking and thinking_effort is not None:
            if thinking_effort not in _KIMI_K3_THINKING_EFFORTS:
                raise ValueError(
                    f"Unsupported thinking_effort={thinking_effort!r}; "
                    "supported values are ['high', 'low', 'max']."
                )

        prompts = []
        for messages in conversations:
            parts = []
            if thinking and thinking_effort is not None:
                parts.extend(
                    [
                        self._open_tag(
                            "message",
                            [("role", "system"), ("type", "thinking-effort")],
                        ),
                        "`thinking_effort` guides on how much to think in your "
                        "thinking channel (not including the response channel), "
                        "supported values include `low`, `medium`, `high`, and "
                        "`max`.\n"
                        "Now the system is invoked with "
                        f"`thinking_effort={thinking_effort}`.",
                        self._close_tag("message"),
                        _KIMI_K3_END_OF_MESSAGE_TOKEN,
                    ]
                )

            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("Kimi K3 chat messages must be dictionaries")
                role = message.get("role")
                if role not in ("system", "user", "assistant"):
                    raise ValueError(
                        f"Kimi K3 role {role!r} is unavailable in the safe "
                        "text-only tokenizer fallback"
                    )

                attrs = [("role", role)]
                if message.get("name"):
                    self._validate_plain_text(str(message["name"]))
                    attrs.append(("name", message["name"]))
                content = message.get("content", "")
                if not isinstance(content, str):
                    raise ValueError(
                        "Kimi K3 multimodal chat content is unavailable in the "
                        "safe text-only tokenizer fallback"
                    )
                self._validate_plain_text(content)

                parts.append(self._open_tag("message", attrs))
                if role == "assistant":
                    if message.get("tool_calls"):
                        raise ValueError(
                            "Kimi K3 tool rendering is unavailable in the safe "
                            "local tokenizer fallback"
                        )
                    if thinking:
                        reasoning = message.get("reasoning_content") or message.get(
                            "reasoning", ""
                        )
                        self._validate_plain_text(str(reasoning))
                        parts.extend(
                            [
                                self._open_tag("think"),
                                str(reasoning),
                                self._close_tag("think"),
                            ]
                        )
                    parts.extend(
                        [
                            self._open_tag("response"),
                            content,
                            self._close_tag("response"),
                        ]
                    )
                else:
                    parts.append(content)
                parts.extend(
                    [
                        self._close_tag("message"),
                        _KIMI_K3_END_OF_MESSAGE_TOKEN,
                    ]
                )

            if add_generation_prompt:
                parts.append(self._open_tag("message", [("role", "assistant")]))
                parts.append(self._open_tag("think" if thinking else "response"))
            prompts.append("".join(parts))

        if not tokenize:
            return prompts if is_batched else prompts[0]

        encoded_inputs = [
            self.encode(prompt, add_special_tokens=False) for prompt in prompts
        ]
        if truncation and max_length is not None:
            encoded_inputs = [ids[:max_length] for ids in encoded_inputs]

        needs_batch_encoding = (
            is_batched or padding or return_tensors is not None or return_dict
        )
        if not needs_batch_encoding:
            return encoded_inputs[0]

        features = [
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
            }
            for ids in encoded_inputs
        ]
        batch = self.pad(
            features,
            padding=padding,
            max_length=max_length if padding else None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        if return_dict:
            return batch
        if is_batched:
            return batch["input_ids"]
        if return_tensors is None:
            return batch["input_ids"][0]
        return batch["input_ids"]


def _read_tokenizer_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fid:
            return json.load(fid)
    except (OSError, JSONDecodeError):
        return None


def _is_local_kimi_k3_tokenizer(model_path):
    model_config = _read_tokenizer_json(model_path / "config.json")
    tokenizer_config = _read_tokenizer_json(model_path / "tokenizer_config.json")
    if not isinstance(model_config, dict) or not isinstance(tokenizer_config, dict):
        return False

    text_config = model_config.get("text_config")
    auto_map_config = tokenizer_config.get("auto_map")
    if not isinstance(text_config, dict) or not isinstance(auto_map_config, dict):
        return False

    auto_map = auto_map_config.get("AutoTokenizer")
    if isinstance(auto_map, (list, tuple)):
        auto_map = auto_map[0] if auto_map else None

    return (
        model_config.get("model_type") == "kimi_k3"
        and text_config.get("model_type") == "kimi_linear"
        and tokenizer_config.get("tokenizer_class") == "TikTokenTokenizer"
        and auto_map == "tokenization_kimi.TikTokenTokenizer"
        and (model_path / "tiktoken.model").is_file()
    )


def _load_local_tiktoken_ranks(vocab_file):
    ranks = {}
    rank_values = set()
    with open(vocab_file, "rb") as fid:
        for line_number, line in enumerate(fid, start=1):
            if not line.strip():
                continue
            try:
                encoded_token, rank_text = line.split()
                token = base64.b64decode(encoded_token, validate=True)
                rank = int(rank_text)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid tiktoken.model entry on line {line_number}"
                ) from error

            if not token or token in ranks or rank in rank_values:
                raise ValueError(
                    f"Duplicate or empty tiktoken.model entry on line {line_number}"
                )
            ranks[token] = rank
            rank_values.add(rank)

    if rank_values != set(range(len(ranks))):
        raise ValueError("tiktoken.model ranks must be contiguous from zero")
    return ranks


def _build_local_kimi_k3_tokenizer(model_path, tokenizer_config_extra=None):
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        pre_tokenizers,
        processors,
    )
    from tokenizers.models import BPE
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    model_config = _read_tokenizer_json(model_path / "config.json")
    tokenizer_config = _read_tokenizer_json(model_path / "tokenizer_config.json")
    if not isinstance(model_config, dict) or not isinstance(tokenizer_config, dict):
        raise ValueError("Kimi K3 tokenizer configuration is missing or invalid")

    mergeable_ranks = _load_local_tiktoken_ranks(model_path / "tiktoken.model")
    num_base_tokens = len(mergeable_ranks)
    expected_vocab_size = num_base_tokens + _KIMI_K3_NUM_RESERVED_TOKENS
    if model_config.get("text_config", {}).get("vocab_size") != expected_vocab_size:
        raise ValueError(
            "Kimi K3 tokenizer vocabulary does not match text_config.vocab_size"
        )

    added_tokens_decoder = tokenizer_config.get("added_tokens_decoder", {})
    try:
        added_tokens_decoder = {
            int(token_id): metadata
            for token_id, metadata in added_tokens_decoder.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Kimi K3 added_tokens_decoder is invalid") from error

    first_reserved_id = num_base_tokens
    last_reserved_id = expected_vocab_size
    if any(
        token_id < first_reserved_id or token_id >= last_reserved_id
        for token_id in added_tokens_decoder
    ):
        raise ValueError("Kimi K3 added token ID is outside the reserved range")

    byte_encoder = bytes_to_unicode()

    def token_bytes_to_string(token):
        return "".join(byte_encoder[byte] for byte in token)

    vocab = {
        token_bytes_to_string(token): rank for token, rank in mergeable_ranks.items()
    }
    added_tokens = []
    for token_id in range(first_reserved_id, last_reserved_id):
        metadata = added_tokens_decoder.get(token_id, {})
        content = metadata.get("content", f"<|reserved_token_{token_id}|>")
        if content in vocab:
            raise ValueError(f"Duplicate Kimi K3 token content {content!r}")
        vocab[content] = token_id
        added_tokens.append(
            AddedToken(
                content,
                single_word=metadata.get("single_word", False),
                lstrip=metadata.get("lstrip", False),
                rstrip=metadata.get("rstrip", False),
                normalized=metadata.get("normalized", False),
                special=metadata.get("special", False),
            )
        )

    merges_with_ranks = []
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        local_merges = []
        for split_index in range(1, len(token)):
            left, right = token[:split_index], token[split_index:]
            if left in mergeable_ranks and right in mergeable_ranks:
                local_merges.append(
                    (
                        mergeable_ranks[left],
                        mergeable_ranks[right],
                        left,
                        right,
                    )
                )
        for _, _, left, right in sorted(local_merges):
            merges_with_ranks.append((rank, left, right))

    merges = [
        (token_bytes_to_string(left), token_bytes_to_string(right))
        for _, left, right in sorted(merges_with_ranks, key=lambda item: item[0])
    ]
    backend = Tokenizer(BPE(vocab, merges, fuse_unk=False))
    if hasattr(backend.model, "ignore_merges"):
        backend.model.ignore_merges = True
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(_KIMI_K3_PATTERN),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    backend.add_tokens(added_tokens)

    init_kwargs = {
        "tokenizer_object": backend,
        "bos_token": tokenizer_config.get("bos_token"),
        "eos_token": tokenizer_config.get("eos_token"),
        "unk_token": tokenizer_config.get("unk_token"),
        "pad_token": tokenizer_config.get("pad_token"),
        "additional_special_tokens": tokenizer_config.get(
            "additional_special_tokens", []
        ),
        "clean_up_tokenization_spaces": tokenizer_config.get(
            "clean_up_tokenization_spaces", False
        ),
        "model_max_length": tokenizer_config.get("model_max_length"),
    }
    tokenizer_config_extra = tokenizer_config_extra or {}
    for key in (
        "padding_side",
        "truncation_side",
        "model_max_length",
        "clean_up_tokenization_spaces",
    ):
        if key in tokenizer_config_extra:
            init_kwargs[key] = tokenizer_config_extra[key]

    tokenizer = _KimiK3TokenizerFast(**init_kwargs)
    if (
        tokenizer.vocab_size != expected_vocab_size
        or len(tokenizer) != expected_vocab_size
    ):
        raise ValueError(
            "Kimi K3 fast tokenizer vocabulary was constructed incorrectly"
        )
    return tokenizer


def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    return isinstance(decoder, dict) and decoder.get("type", None) == "ByteLevel"


def _infer_tool_parser(chat_template):
    """Attempt to auto-infer a tool parser from the chat template."""
    if not isinstance(chat_template, str):
        return None
    elif "<minimax:tool_call>" in chat_template:
        return "minimax_m2"
    elif "<|tool_call>" in chat_template and "<tool_call|>" in chat_template:
        return "gemma4"
    elif "<start_function_call>" in chat_template:
        return "function_gemma"
    elif "<longcat_tool_call>" in chat_template:
        return "longcat"
    elif "<arg_key>" in chat_template:
        return "glm47"
    elif "<|tool_list_start|>" in chat_template:
        return "pythonic"
    elif (
        "<tool_call>\\n<function=" in chat_template
        or "<tool_call>\n<function=" in chat_template
    ):
        return "qwen3_coder"
    elif "<|tool_calls_section_begin|>" in chat_template:
        return "kimi_k2"
    elif "[TOOL_CALLS]" in chat_template:
        return "mistral"
    elif "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return "json_tools"
    return None


def load(
    model_path,
    tokenizer_config_extra: Optional[Dict[str, Any]] = None,
    eos_token_ids=None,
) -> TokenizerWrapper:
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = NaiveStreamingDetokenizer

    tokenizer_file = model_path / "tokenizer.json"

    if tokenizer_file.exists():
        with open(tokenizer_file, "r", encoding="utf-8") as fid:
            try:
                tokenizer_content = json.load(fid)
            except JSONDecodeError as e:
                raise JSONDecodeError("Failed to parse tokenizer.json", e.doc, e.pos)

        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    tokenizer_config_file = model_path / "tokenizer_config.json"
    chat_template = None

    tokenizer_config_extra = tokenizer_config_extra or {}
    is_local_kimi_k3 = _is_local_kimi_k3_tokenizer(model_path)
    use_safe_kimi_k3 = (
        is_local_kimi_k3
        and tokenizer_config_extra.get("trust_remote_code") is not True
    )
    if use_safe_kimi_k3:
        warnings.warn(
            "Loading Kimi K3's tokenizer from local data without executing "
            "the checkpoint's custom Python. The safe fallback supports "
            "text-only XTML chat rendering; tools and media are unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        tokenizer = _build_local_kimi_k3_tokenizer(model_path, tokenizer_config_extra)
        detokenizer_class = partial(
            BPEStreamingDetokenizer,
            trim_initial_space=False,
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, **tokenizer_config_extra
            )
        except (AttributeError, ValueError) as e:
            # Transformers may not recognize brand-new model_types (e.g.
            # deepseek_v4 before a transformers release adds it). Fall back
            # to a generic tokenizer built from tokenizer.json in the repo.
            if "config" in tokenizer_config_extra:
                raise
            from transformers import PretrainedConfig

            stub_kwargs: Dict[str, Any] = {}
            model_config_file = model_path / "config.json"
            if model_config_file.exists():
                try:
                    with open(model_config_file, "r") as f:
                        raw = json.load(f)
                    for key in (
                        "model_type",
                        "max_position_embeddings",
                        "vocab_size",
                        "bos_token_id",
                        "eos_token_id",
                        "pad_token_id",
                    ):
                        if key in raw:
                            stub_kwargs[key] = raw[key]
                except (OSError, JSONDecodeError):
                    pass

            warnings.warn(
                "Falling back to a generic tokenizer because Transformers "
                f"does not recognize this model config yet: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                config=PretrainedConfig(**stub_kwargs),
                **tokenizer_config_extra,
            )

    if is_local_kimi_k3 and eos_token_ids is None:
        model_config = _read_tokenizer_json(model_path / "config.json")
        configured_eos = model_config.get("eos_token_id")
        if configured_eos is None:
            configured_eos = model_config.get("text_config", {}).get("eos_token_id")
        if isinstance(configured_eos, int):
            configured_eos = [configured_eos]
        elif configured_eos is not None:
            if not isinstance(configured_eos, (list, tuple)) or not all(
                isinstance(token_id, int) for token_id in configured_eos
            ):
                raise ValueError("Kimi K3 eos_token_id must contain integer IDs")
            configured_eos = list(configured_eos)
        if configured_eos is not None and any(
            token_id < 0 or token_id >= len(tokenizer)
            for token_id in configured_eos
        ):
            raise ValueError("Kimi K3 eos_token_id is outside the vocabulary")
        eos_token_ids = configured_eos

    tokenizer_config = tokenizer.init_kwargs

    if chat_template_type := tokenizer_config.get("chat_template_type", False):
        chat_template = importlib.import_module(
            f"mlx_lm.chat_templates.{chat_template_type}"
        ).apply_chat_template

    tool_parser_type = tokenizer_config.get(
        "tool_parser_type", _infer_tool_parser(tokenizer.chat_template)
    )

    if tool_parser_type is not None:
        tool_module = importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
        tool_parser = tool_module.parse_tool_call
        tool_call_start = tool_module.tool_call_start
        tool_call_end = tool_module.tool_call_end
        tokenizer_config["tool_parser_type"] = tool_parser_type
    else:
        tool_parser = None
        tool_call_start = None
        tool_call_end = None

    return TokenizerWrapper(
        tokenizer,
        detokenizer_class,
        eos_token_ids=eos_token_ids,
        chat_template=chat_template,
        tool_parser=tool_parser,
        tool_call_start=tool_call_start,
        tool_call_end=tool_call_end,
        thinking_markers=(
            ("<|open|>think<|sep|>", "<|close|>think<|sep|>")
            if is_local_kimi_k3
            else None
        ),
    )


def no_bos_or_eos(sequence: List, bos: int, eos: int) -> List:
    removed_bos = sequence if sequence[0] != bos else sequence[1:]
    return removed_bos[:-1] if removed_bos[-1] == eos else removed_bos
