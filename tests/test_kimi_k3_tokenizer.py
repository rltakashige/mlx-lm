import base64
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    _build_local_kimi_k3_tokenizer,
    _is_local_kimi_k3_tokenizer,
    _load_local_tiktoken_ranks,
    load,
)


def _token_metadata(content, special):
    return {
        "content": content,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": special,
    }


def _write_kimi_k3_tokenizer_fixture(model_path):
    ranks = [(bytes([value]), value) for value in range(256)]
    ranks.extend([(b"hi", 256), (b"him", 257)])
    with open(model_path / "tiktoken.model", "wb") as fid:
        for token, rank in ranks:
            fid.write(base64.b64encode(token) + f" {rank}\n".encode())

    first_reserved_id = len(ranks)
    known_tokens = {
        first_reserved_id: ("[BOS]", True),
        first_reserved_id + 1: ("[EOS]", True),
        first_reserved_id + 2: ("<|end_of_msg|>", True),
        first_reserved_id + 3: ("<|open|>", False),
        first_reserved_id + 4: ("<|close|>", False),
        first_reserved_id + 5: ("<|sep|>", False),
        first_reserved_id + 6: ("[start_header_id]", True),
        first_reserved_id + 7: ("[end_header_id]", True),
        first_reserved_id + 9: ("[EOT]", True),
        first_reserved_id + 18: ("<|media_begin|>", True),
        first_reserved_id + 19: ("<|media_content|>", True),
        first_reserved_id + 20: ("<|media_end|>", True),
        first_reserved_id + 21: ("<|media_pad|>", True),
        first_reserved_id + 65: ("<osagent_mode>", True),
        first_reserved_id + 254: ("[UNK]", True),
        first_reserved_id + 255: ("[PAD]", True),
    }
    tokenizer_config = {
        "added_tokens_decoder": {
            str(token_id): _token_metadata(content, special)
            for token_id, (content, special) in known_tokens.items()
        },
        "additional_special_tokens": [
            "<|end_of_msg|>",
            "[start_header_id]",
            "[end_header_id]",
            "[EOT]",
            "<|media_begin|>",
            "<|media_content|>",
            "<|media_end|>",
            "<|media_pad|>",
            "<osagent_mode>",
        ],
        "auto_map": {
            "AutoTokenizer": [
                "tokenization_kimi.TikTokenTokenizer",
                None,
            ]
        },
        "bos_token": "[BOS]",
        "clean_up_tokenization_spaces": False,
        "eos_token": "[EOS]",
        "model_max_length": 1024,
        "pad_token": "[PAD]",
        "tokenizer_class": "TikTokenTokenizer",
        "unk_token": "[UNK]",
    }
    (model_path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["KimiK3ForConditionalGeneration"],
                "bos_token_id": first_reserved_id,
                "eos_token_id": first_reserved_id + 2,
                "model_type": "kimi_k3",
                "pad_token_id": first_reserved_id + 255,
                "text_config": {
                    "model_type": "kimi_linear",
                    "vocab_size": first_reserved_id + 256,
                },
            }
        ),
        encoding="utf-8",
    )
    return first_reserved_id


class TestKimiK3Tokenizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name)
        self.first_reserved_id = _write_kimi_k3_tokenizer_fixture(self.model_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_safe_loader_bypasses_auto_tokenizer(self):
        self.assertTrue(_is_local_kimi_k3_tokenizer(self.model_path))
        with mock.patch(
            "mlx_lm.tokenizer_utils.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("AutoTokenizer must not run"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tokenizer = load(
                    self.model_path,
                    {
                        "padding_side": "left",
                        "trust_remote_code": True,
                    },
                )

        self.assertTrue(
            any(
                issubclass(item.category, RuntimeWarning)
                and "without executing" in str(item.message)
                for item in caught
            )
        )
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.vocab_size, self.first_reserved_id + 256)
        self.assertEqual(len(tokenizer._tokenizer), self.first_reserved_id + 256)
        self.assertTrue(tokenizer.has_chat_template)
        self.assertIsNone(tokenizer.chat_template)
        self.assertTrue(tokenizer.has_thinking)
        self.assertEqual(tokenizer.think_start, "<|open|>think<|sep|>")
        self.assertEqual(tokenizer.think_end, "<|close|>think<|sep|>")
        self.assertEqual(
            tokenizer.eos_token_ids,
            {self.first_reserved_id + 2},
        )
        self.assertIsInstance(
            tokenizer.detokenizer,
            BPEStreamingDetokenizer,
        )

    def test_explicit_eos_override_wins_over_model_config(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tokenizer = load(
                self.model_path,
                eos_token_ids=[self.first_reserved_id + 1],
            )
        self.assertEqual(tokenizer.eos_token_ids, {self.first_reserved_id + 1})

    def test_encode_decode_and_special_ids_match_reference(self):
        tokenizer = _build_local_kimi_k3_tokenizer(self.model_path)
        bos_id = self.first_reserved_id
        end_of_message_id = self.first_reserved_id + 2
        open_id = self.first_reserved_id + 3
        pad_id = self.first_reserved_id + 255

        self.assertEqual(
            tokenizer.encode("him!", add_special_tokens=False),
            [257, ord("!")],
        )
        self.assertEqual(
            tokenizer.encode(
                "[BOS]him!<|end_of_msg|><|open|>[PAD]",
                add_special_tokens=False,
            ),
            [bos_id, 257, ord("!"), end_of_message_id, open_id, pad_id],
        )
        self.assertEqual(
            tokenizer.encode("你好🙂", add_special_tokens=False),
            list("你好🙂".encode()),
        )
        self.assertEqual(
            tokenizer.encode(
                f"<|reserved_token_{self.first_reserved_id + 8}|>",
                add_special_tokens=False,
            ),
            [self.first_reserved_id + 8],
        )

        ids = [bos_id, 257, ord("!"), end_of_message_id, open_id, pad_id]
        self.assertEqual(
            tokenizer.decode(ids, skip_special_tokens=False),
            "[BOS]him!<|end_of_msg|><|open|>[PAD]",
        )
        self.assertEqual(
            tokenizer.decode(ids, skip_special_tokens=True),
            "him!<|open|>",
        )
        self.assertNotIn(open_id, tokenizer.all_special_ids)
        self.assertIn(end_of_message_id, tokenizer.all_special_ids)

    def test_streaming_decode_matches_decode(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tokenizer = load(self.model_path)

        cases = [
            "hi! 你好🙂",
            " leading space",
            "\x00",
            "line one\nline two\r\n",
            ("<|open|>think<|sep|>reasoning" "<|close|>think<|sep|><|end_of_msg|>"),
            "🙂\n <|open|>",
        ]
        for text in cases:
            with self.subTest(text=repr(text)):
                ids = tokenizer.encode(text, add_special_tokens=False)
                detokenizer = tokenizer.detokenizer
                segments = []
                for token_id in ids:
                    detokenizer.add_token(token_id)
                    segments.append(detokenizer.last_segment)
                detokenizer.finalize()
                segments.append(detokenizer.last_segment)
                expected = tokenizer.decode(ids)
                self.assertEqual("".join(segments), expected)
                self.assertEqual(detokenizer.text, expected)
                self.assertEqual(detokenizer.tokens, ids)

    def test_text_chat_template_matches_k3_xtml(self):
        tokenizer = _build_local_kimi_k3_tokenizer(self.model_path)
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ]
        thinking_prefix = (
            '<|open|>message role="system" type="thinking-effort"<|sep|>'
            "`thinking_effort` guides on how much to think in your thinking "
            "channel (not including the response channel), supported values "
            "include `low`, `medium`, `high`, and `max`.\n"
            "Now the system is invoked with `thinking_effort=max`."
            "<|close|>message<|sep|><|end_of_msg|>"
        )
        conversation = (
            '<|open|>message role="system"<|sep|>Be concise.'
            "<|close|>message<|sep|><|end_of_msg|>"
            '<|open|>message role="user"<|sep|>hello'
            "<|close|>message<|sep|><|end_of_msg|>"
            '<|open|>message role="assistant"<|sep|>'
            "<|open|>think<|sep|>"
        )
        expected = thinking_prefix + conversation
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        self.assertEqual(rendered, expected)
        self.assertEqual(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
            tokenizer.encode(expected, add_special_tokens=False),
        )

    def test_batched_chat_template_and_output_controls_match_reference_contract(self):
        tokenizer = _build_local_kimi_k3_tokenizer(self.model_path)
        conversations = [
            [{"role": "user", "content": "short"}],
            [{"role": "user", "content": "a longer prompt"}],
        ]
        expected_prompts = [
            tokenizer.apply_chat_template(conversation, tokenize=False)
            for conversation in conversations
        ]
        self.assertEqual(
            tokenizer.apply_chat_template(conversations, tokenize=False),
            expected_prompts,
        )

        expected_ids = [
            tokenizer.encode(prompt, add_special_tokens=False)
            for prompt in expected_prompts
        ]
        batch = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            padding=True,
            return_tensors="np",
            return_dict=True,
        )
        self.assertEqual(
            batch["input_ids"].shape,
            (2, max(map(len, expected_ids))),
        )
        for row, ids in enumerate(expected_ids):
            np.testing.assert_array_equal(
                batch["input_ids"][row, : len(ids)],
                ids,
            )
            np.testing.assert_array_equal(
                batch["attention_mask"][row, : len(ids)],
                1,
            )
            np.testing.assert_array_equal(
                batch["attention_mask"][row, len(ids) :],
                0,
            )

        truncated = tokenizer.apply_chat_template(
            conversations[0],
            tokenize=True,
            truncation=True,
            max_length=17,
            return_tensors="np",
            return_dict=True,
        )
        self.assertEqual(truncated["input_ids"].shape, (1, 17))
        np.testing.assert_array_equal(
            truncated["input_ids"][0],
            expected_ids[0][:17],
        )
        np.testing.assert_array_equal(truncated["attention_mask"], 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wrapped = load(self.model_path)
        wrapped_batch = wrapped.apply_chat_template(
            conversations,
            tokenize=True,
            padding=True,
            return_tensors="np",
        )
        np.testing.assert_array_equal(wrapped_batch, batch["input_ids"])

    def test_thinking_effort_matches_k3_defaults(self):
        tokenizer = _build_local_kimi_k3_tokenizer(self.model_path)
        messages = [{"role": "user", "content": "hello"}]
        default_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        self.assertIn('type="thinking-effort"', default_prompt)
        self.assertIn("`thinking_effort=max`", default_prompt)

        for effort in ("low", "high", "max"):
            prompt = tokenizer.apply_chat_template(
                messages,
                thinking_effort=effort,
                tokenize=False,
            )
            self.assertIn(f"`thinking_effort={effort}`", prompt)

        no_effort_prompt = tokenizer.apply_chat_template(
            messages,
            thinking_effort=None,
            tokenize=False,
        )
        self.assertNotIn('type="thinking-effort"', no_effort_prompt)
        self.assertIn("<|open|>think<|sep|>", no_effort_prompt)

        no_thinking_prompt = tokenizer.apply_chat_template(
            messages,
            thinking=False,
            tokenize=False,
        )
        self.assertNotIn('type="thinking-effort"', no_thinking_prompt)
        self.assertNotIn("<|open|>think<|sep|>", no_thinking_prompt)
        self.assertTrue(no_thinking_prompt.endswith("<|open|>response<|sep|>"))

        no_thinking_none_prompt = tokenizer.apply_chat_template(
            messages,
            thinking=None,
            tokenize=False,
        )
        self.assertEqual(no_thinking_none_prompt, no_thinking_prompt)

        for effort in ("medium", "invalid"):
            with self.assertRaisesRegex(ValueError, "Unsupported thinking_effort"):
                tokenizer.apply_chat_template(
                    messages,
                    thinking_effort=effort,
                    tokenize=False,
                )

    def test_reasoning_effort_alias_matches_k3_renderer(self):
        tokenizer = _build_local_kimi_k3_tokenizer(self.model_path)
        messages = [{"role": "user", "content": "hello"}]

        for effort in ("low", "high", "max"):
            prompt = tokenizer.apply_chat_template(
                messages,
                reasoning_effort=effort,
                tokenize=False,
            )
            self.assertIn(f"`thinking_effort={effort}`", prompt)

        prompt = tokenizer.apply_chat_template(
            messages,
            reasoning_effort="low",
            thinking_effort="high",
            tokenize=False,
        )
        self.assertIn("`thinking_effort=high`", prompt)
        self.assertNotIn("`thinking_effort=low`", prompt)

        no_thinking_prompt = tokenizer.apply_chat_template(
            messages,
            reasoning_effort="none",
            tokenize=False,
        )
        self.assertNotIn('type="thinking-effort"', no_thinking_prompt)
        self.assertNotIn("<|open|>think<|sep|>", no_thinking_prompt)
        self.assertTrue(no_thinking_prompt.endswith("<|open|>response<|sep|>"))

        self.assertEqual(
            tokenizer.apply_chat_template(
                messages,
                reasoning_effort=None,
                tokenize=False,
            ),
            tokenizer.apply_chat_template(messages, tokenize=False),
        )
        self.assertEqual(
            tokenizer.apply_chat_template(
                messages,
                reasoning_effort="none",
                enable_thinking=None,
                tokenize=False,
            ),
            no_thinking_prompt,
        )

        for effort in ("minimal", "medium", "xhigh", "invalid"):
            with self.assertRaisesRegex(ValueError, "Unsupported thinking_effort"):
                tokenizer.apply_chat_template(
                    messages,
                    reasoning_effort=effort,
                    tokenize=False,
                )

        with self.assertRaisesRegex(ValueError, "Unsupported thinking_effort"):
            tokenizer.apply_chat_template(
                messages,
                reasoning_effort="none",
                enable_thinking=True,
                tokenize=False,
            )

    def test_chat_template_fails_closed_for_tools_and_multimodal(self):
        tokenizer = _build_local_kimi_k3_tokenizer(self.model_path)
        with self.assertRaisesRegex(ValueError, "tool rendering is unavailable"):
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}],
                tools=[{"type": "function"}],
            )
        with self.assertRaisesRegex(ValueError, "multimodal chat content"):
            tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
            )
        with self.assertRaisesRegex(ValueError, "reserved control token"):
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "literal <|open|> marker"}]
            )
        with self.assertRaisesRegex(ValueError, "continue_final_message"):
            tokenizer.apply_chat_template(
                [{"role": "assistant", "content": "partial"}],
                add_generation_prompt=False,
                continue_final_message=True,
            )

    def test_invalid_rank_file_is_rejected(self):
        invalid_file = self.model_path / "invalid.tiktoken"
        invalid_file.write_bytes(b"YQ== 0\nYg== 0\n")
        with self.assertRaisesRegex(ValueError, "Duplicate or empty"):
            _load_local_tiktoken_ranks(invalid_file)


if __name__ == "__main__":
    unittest.main()
