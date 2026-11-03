# Local AI Chat

A lightweight, local chat interface powered by `llama.cpp` through
`llama-cpp-python` and a Gradio web UI. The default configuration targets a
4-vCPU, 8-GB-RAM CPU host and downloads the Qwen2.5 3B Q4_K_M GGUF on the first
chat request.

## Requirements

- Python 3.10 or newer
- A 64-bit operating system
- Enough free disk space for the GGUF model (about 2 GB)
- On systems without a compatible prebuilt `llama-cpp-python` wheel, a C/C++
  compiler and CMake to build its CPU backend

## Setup

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and configure the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to change the defaults. The default
inference settings are deliberately fixed to `N_THREADS=4` and `N_CTX=4096`;
the app rejects a thread count other than four. `MODEL_PATH` controls where the
GGUF file is stored, while `HOST` and `PORT` configure the Gradio listener.
Keep the default host (`127.0.0.1`) for local-only access.

## Run

```bash
python app.py
```

Open the local URL printed by Gradio (by default
`http://127.0.0.1:7860`). The model is downloaded when the first message is
sent, then kept loaded and reused for later turns. Closing the application
releases the model and its context. Use the sidebar to adjust the system prompt,
temperature, top-p, and maximum output length. The chat includes clear and
regenerate controls, and each response has a copy button.

## Model and memory notes

The default Hugging Face repository is `Qwen/Qwen2.5-3B-Instruct-GGUF`, with
`qwen2.5-3b-instruct-q4_k_m.gguf`. Hugging Face Hub uses its normal cache and
authentication settings; set `HF_TOKEN` in `.env` only if repository access
requires it. Model availability and license terms are determined by the model
publisher.

Inference runs on CPU (`n_gpu_layers=0`), with four inference and batch threads,
a 4096-token context, and a 256-token inference batch. The model is initialized
once and reused; simultaneous generations are serialized so the single model
context is not accessed concurrently. The selected GGUF quantization and
context still require memory; close other memory-heavy processes if loading
fails. This project does not start a separate API server: the Python process
serves the Gradio UI and uses the llama.cpp Python binding directly.

Before each generation, the app tokenizes the Qwen chat-formatted prompt with
the loaded model, reserves `Max tokens` from the 4096-token window, and keeps
the system prompt, current user message, and newest complete conversation turns
that fit. If the required system prompt and current message alone exceed the
remaining context, the UI reports that they need to be shortened or the output
limit reduced. If a generation is cancelled, the worker stops at the next
generated token and closes its stream before releasing the model lock.

## Checks

Run the dependency-light unit tests with:

```bash
python -m unittest discover -s tests
```

They exercise prompt budgeting, cancellation cleanup, and invalid integer
settings without downloading model weights.
