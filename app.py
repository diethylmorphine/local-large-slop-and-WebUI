"""Gradio entry point for the local Qwen GGUF chat interface."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import gradio as gr

from config import AppConfig, ConfigurationError
from model_manager import InferenceError, ModelLoadError, ModelManager

CSS = """
:root, body { color-scheme: dark; }
body, .gradio-container { background: #0b1020 !important; color: #e5e7eb !important; }
.gradio-container { max-width: 1440px !important; }
footer { display: none !important; }
"""

try:
    config = AppConfig.from_env()
except ConfigurationError as exc:
    raise SystemExit(f"Configuration error: {exc}") from None
model_manager = ModelManager(config)


class ContextBudgetError(ValueError):
    """Raised when the required system and current user prompts cannot fit."""


def _copy_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in (history or [])
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]


def _format_prompt(messages: list[dict[str, str]]) -> str:
    serialized = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )
    return f"{serialized}<|im_start|>assistant\n"


async def _fit_context(
    conversation: list[dict[str, str]],
    system_prompt: str,
    max_tokens: int,
    n_ctx: int,
    count_tokens: Callable[[str], Awaitable[int]],
) -> list[dict[str, str]]:
    if max_tokens <= 0 or max_tokens >= n_ctx:
        raise ContextBudgetError("Max tokens must be positive and smaller than the context window.")
    if not conversation or conversation[-1]["role"] != "user":
        raise ContextBudgetError("The conversation must end with the current user message.")

    system = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
    current_user = conversation[-1]
    prior = conversation[:-1]
    complete_turns: list[list[dict[str, str]]] = []
    for index in range(0, len(prior) - 1, 2):
        user, assistant = prior[index : index + 2]
        if user["role"] == "user" and assistant["role"] == "assistant":
            complete_turns.append([user, assistant])

    async def fits(messages: list[dict[str, str]]) -> bool:
        token_count = await count_tokens(_format_prompt(messages))
        return token_count + max_tokens <= n_ctx

    selected_turns: list[list[dict[str, str]]] = []
    required_messages = system + [current_user]
    if not await fits(required_messages):
        raise ContextBudgetError(
            "The system prompt and current message exceed the available context. "
            "Shorten them or reduce Max tokens."
        )

    for turn in reversed(complete_turns):
        candidate_turns = [turn, *selected_turns]
        candidate = system + [message for item in candidate_turns for message in item]
        candidate.append(current_user)
        if not await fits(candidate):
            break
        selected_turns = candidate_turns

    return system + [message for turn in selected_turns for message in turn] + [current_user]


async def _generate_turn(
    conversation: list[dict[str, str]],
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> AsyncIterator[list[dict[str, str]]]:
    response = ""
    max_tokens = int(max_tokens)
    try:
        messages = await _fit_context(
            conversation=conversation,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            n_ctx=config.n_ctx,
            count_tokens=model_manager.count_tokens,
        )
        async for part in model_manager.stream_chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            response += part
            yield conversation + [{"role": "assistant", "content": response}]
        if not response:
            yield conversation + [{"role": "assistant", "content": "The model returned an empty response."}]
    except (ContextBudgetError, ModelLoadError, InferenceError) as exc:
        yield conversation + [{"role": "assistant", "content": f"**Model error:** {exc}"}]


async def send_message(
    message: str,
    history: list[dict[str, str]] | None,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> AsyncIterator[tuple[list[dict[str, str]], str]]:
    conversation = _copy_history(history)
    text = message.strip()
    if not text:
        yield conversation, ""
        return

    conversation.append({"role": "user", "content": text})
    async for updated in _generate_turn(conversation, system_prompt, temperature, top_p, max_tokens):
        yield updated, ""


async def regenerate(
    history: list[dict[str, str]] | None,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    draft: str,
) -> AsyncIterator[tuple[list[dict[str, str]], str]]:
    conversation = _copy_history(history)
    if conversation and conversation[-1]["role"] == "assistant":
        conversation.pop()
    if not conversation or conversation[-1]["role"] != "user":
        yield conversation, draft
        return

    async for updated in _generate_turn(conversation, system_prompt, temperature, top_p, max_tokens):
        yield updated, draft


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Local AI Chat") as demo:
        gr.Markdown("# Local AI Chat\nPrivate, CPU-powered chat running on this machine.")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Conversation settings")
                system_prompt = gr.Textbox(
                    label="System prompt",
                    value="You are a helpful assistant.",
                    lines=5,
                )
                temperature = gr.Slider(0, 2, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0, 1, value=0.9, step=0.05, label="Top-P")
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
                regenerate_button = gr.Button("Regenerate response")
                clear_button = gr.Button("Clear chat", variant="secondary")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=620,
                    show_copy_button=True,
                    label="Chat",
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Message the local model...",
                        label="Message",
                        scale=5,
                        lines=1,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)

        inputs = [user_input, chatbot, system_prompt, temperature, top_p, max_tokens]
        outputs = [chatbot, user_input]
        send_button.click(send_message, inputs=inputs, outputs=outputs)
        user_input.submit(send_message, inputs=inputs, outputs=outputs)
        regenerate_button.click(
            regenerate,
            inputs=[chatbot, system_prompt, temperature, top_p, max_tokens, user_input],
            outputs=outputs,
        )
        clear_button.click(lambda: ([], ""), inputs=None, outputs=outputs, queue=False)
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(
        server_name=config.host,
        server_port=config.port,
        share=False,
    )
