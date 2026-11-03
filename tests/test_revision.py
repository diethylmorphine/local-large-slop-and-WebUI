from __future__ import annotations

import asyncio
import importlib
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
sys.modules.setdefault("gradio", types.SimpleNamespace())

from app import ContextBudgetError, _fit_context, _format_prompt
from config import AppConfig, ConfigurationError
from model_manager import ModelManager


class ContextBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_system_and_recent_complete_turns(self) -> None:
        conversation = [
            {"role": "user", "content": "x " * 35},
            {"role": "assistant", "content": "y " * 35},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "current question"},
        ]

        async def count_tokens(text: str) -> int:
            return len(text.split())

        result = await _fit_context(
            conversation,
            "always keep this system prompt",
            max_tokens=16,
            n_ctx=55,
            count_tokens=count_tokens,
        )
        prompt = _format_prompt(result)

        self.assertIn("always keep this system prompt", prompt)
        self.assertIn("recent question", prompt)
        self.assertIn("recent answer", prompt)
        self.assertIn("current question", prompt)
        self.assertNotIn("x x x", prompt)
        self.assertLessEqual(await count_tokens(prompt) + 16, 55)

    async def test_rejects_required_messages_that_exceed_context(self) -> None:
        async def count_tokens(text: str) -> int:
            return len(text.split())

        with self.assertRaisesRegex(ContextBudgetError, "exceed the available context"):
            await _fit_context(
                [{"role": "user", "content": "word " * 40}],
                "system prompt",
                max_tokens=16,
                n_ctx=32,
                count_tokens=count_tokens,
            )


class ConfigurationTests(unittest.TestCase):
    def test_malformed_integer_settings_have_explicit_errors(self) -> None:
        with patch.dict(os.environ, {"PORT": "not-a-port"}, clear=True):
            with self.assertRaisesRegex(
                ConfigurationError, "PORT must be an integer; received 'not-a-port'"
            ):
                AppConfig.from_env()

    def test_app_reports_configuration_error_without_traceback(self) -> None:
        import app

        with patch.dict(os.environ, {"PORT": "bad"}, clear=True):
            with self.assertRaisesRegex(
                SystemExit, "Configuration error: PORT must be an integer"
            ):
                importlib.reload(app)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_count_uses_loaded_model_tokenizer(self) -> None:
        class FakeModel:
            def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
                self.options = (text, add_bos, special)
                return [1, 2, 3]

        config = AppConfig(
            model_repo_id="test/repo",
            model_filename="model.gguf",
            model_path=Path("model.gguf"),
            n_threads=4,
            n_ctx=4096,
            host="127.0.0.1",
            port=7860,
            hf_token=None,
        )
        manager = ModelManager(config)
        fake_model = FakeModel()
        manager._model = fake_model

        self.assertEqual(await manager.count_tokens("hello"), 3)
        self.assertEqual(fake_model.options, (b"hello", True, True))
        self.assertFalse(manager._generation_lock.locked())

    async def test_cancelling_stream_closes_worker_and_releases_model_lock(self) -> None:
        closed = threading.Event()

        class FakeModel:
            calls = 0

            def create_chat_completion(self, **kwargs):
                self.calls += 1
                if self.calls > 1:
                    return iter([{"choices": [{"delta": {"content": "next request"}}]}])

                def chunks():
                    try:
                        while True:
                            time.sleep(0.005)
                            yield {"choices": [{"delta": {"content": "token"}}]}
                    finally:
                        closed.set()

                return chunks()

        config = AppConfig(
            model_repo_id="test/repo",
            model_filename="model.gguf",
            model_path=Path("model.gguf"),
            n_threads=4,
            n_ctx=4096,
            host="127.0.0.1",
            port=7860,
            hf_token=None,
        )
        manager = ModelManager(config)
        manager._model = FakeModel()
        stream = manager.stream_chat([{"role": "user", "content": "hello"}], 0.7, 0.9, 64)

        self.assertEqual(await anext(stream), "token")
        await asyncio.wait_for(stream.aclose(), timeout=2)

        self.assertTrue(closed.wait(timeout=1))
        self.assertFalse(manager._generation_lock.locked())
        self.assertFalse(
            any(
                thread.name == "llama-cpp-generation" and thread.is_alive()
                for thread in threading.enumerate()
            )
        )
        next_response = [
            part
            async for part in manager.stream_chat(
                [{"role": "user", "content": "again"}], 0.7, 0.9, 64
            )
        ]
        self.assertEqual(next_response, ["next request"])


if __name__ == "__main__":
    unittest.main()
