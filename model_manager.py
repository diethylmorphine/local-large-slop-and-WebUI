"""Download and reuse a CPU-backed llama.cpp model for chat generation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from config import AppConfig

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the model cannot be downloaded or loaded."""


class InferenceError(RuntimeError):
    """Raised when model generation fails."""


class ModelManager:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: Llama | None = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def load(self) -> Llama:
        """Load the model once and reuse its context for subsequent requests."""
        with self._load_lock:
            if self._model is not None:
                return self._model

            try:
                from llama_cpp import Llama
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ModelLoadError(
                    "Model dependencies are missing. Install them with "
                    "`python -m pip install -r requirements.txt`."
                ) from exc

            try:
                path = self._config.model_path
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    downloaded_path = hf_hub_download(
                        repo_id=self._config.model_repo_id,
                        filename=self._config.model_filename,
                        local_dir=str(path.parent),
                        token=self._config.hf_token,
                    )
                    path = Path(downloaded_path)
                if not path.is_file():
                    raise FileNotFoundError(f"GGUF file was not created at {path}.")
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not download or locate the GGUF model "
                    f"({self._config.model_repo_id}/{self._config.model_filename}): {exc}"
                ) from exc

            try:
                self._model = Llama(
                    model_path=str(path),
                    n_threads=self._config.n_threads,
                    n_threads_batch=self._config.n_threads,
                    n_ctx=self._config.n_ctx,
                    n_batch=256,
                    n_gpu_layers=0,
                    verbose=False,
                )
            except (MemoryError, OSError) as exc:
                raise ModelLoadError(
                    "The model could not allocate its CPU memory. Close other memory-heavy "
                    "applications or choose a smaller GGUF model."
                ) from exc
            except Exception as exc:
                detail = str(exc).lower()
                if "memory" in detail or "alloc" in detail:
                    message = (
                        "The model could not allocate its CPU memory. Close other "
                        "memory-heavy applications or choose a smaller GGUF model."
                    )
                else:
                    message = f"Could not initialize the GGUF model: {exc}"
                raise ModelLoadError(message) from exc

            return self._model

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield generated text without blocking the UI's asyncio event loop."""
        model = await asyncio.to_thread(self.load)
        await self._acquire_generation_lock()

        loop = asyncio.get_running_loop()
        events: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=8)
        stop_event = threading.Event()
        worker_done = threading.Event()

        def publish(kind: str, value: object) -> bool:
            while not stop_event.is_set():
                try:
                    pending_put = asyncio.run_coroutine_threadsafe(
                        events.put((kind, value)), loop
                    )
                except RuntimeError:
                    return False
                try:
                    pending_put.result(timeout=0.1)
                    return True
                except concurrent.futures.TimeoutError:
                    pending_put.cancel()
                except (concurrent.futures.CancelledError, RuntimeError):
                    return False
            return False

        def generate() -> None:
            stream: object | None = None
            try:
                stream = iter(model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=True,
                ))
                while not stop_event.is_set():
                    try:
                        chunk = next(stream)
                    except StopIteration:
                        break
                    text = self._content_from_chunk(chunk)
                    if text and not publish("text", text):
                        break
            except MemoryError:
                publish(
                    "error",
                    InferenceError(
                        "Generation ran out of memory. Try a shorter prompt or fewer max tokens."
                    ),
                )
            except Exception as exc:
                publish("error", InferenceError(f"Generation failed: {exc}"))
            finally:
                try:
                    close = getattr(stream, "close", None)
                    if stop_event.is_set() and callable(close):
                        close()
                except Exception:
                    logger.exception("Could not close the cancelled model stream.")
                finally:
                    self._generation_lock.release()
                    worker_done.set()
                if not stop_event.is_set():
                    publish("done", None)

        try:
            threading.Thread(target=generate, name="llama-cpp-generation", daemon=True).start()
        except Exception:
            self._generation_lock.release()
            raise

        try:
            while True:
                kind, value = await events.get()
                if kind == "text":
                    yield cast(str, value)
                elif kind == "error":
                    if isinstance(value, InferenceError):
                        raise value
                    raise InferenceError("Generation failed with an unexpected error.")
                else:
                    break
        finally:
            stop_event.set()
            await asyncio.to_thread(worker_done.wait)

    async def count_tokens(self, text: str) -> int:
        """Tokenize a prompt with the loaded model while holding its context lock."""
        model = await asyncio.to_thread(self.load)
        await self._acquire_generation_lock()
        try:
            tokenization = asyncio.create_task(
                asyncio.to_thread(
                    model.tokenize,
                    text.encode("utf-8"),
                    add_bos=True,
                    special=True,
                )
            )
            try:
                tokens = await asyncio.shield(tokenization)
            except asyncio.CancelledError:
                await tokenization
                raise
            return len(tokens)
        finally:
            self._generation_lock.release()

    async def _acquire_generation_lock(self) -> None:
        acquire_task = asyncio.create_task(asyncio.to_thread(self._generation_lock.acquire))
        try:
            await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            await acquire_task
            self._generation_lock.release()
            raise

    @staticmethod
    def _content_from_chunk(chunk: object) -> str:
        if not isinstance(chunk, Mapping):
            return ""
        choices = chunk.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return ""
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            return ""
        content = delta.get("content")
        return content if isinstance(content, str) else ""
