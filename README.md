# Local AI Chat

A lightweight, local chat interface powered by `llama.cpp` through
`llama-cpp-python` and a Gradio web UI. The default configuration targets a
4-vCPU, 8-GB-RAM CPU host and downloads the Qwen3 8B Q4_K_M GGUF on the first
chat request. The 5.03-GB model uses `/no_think` by default to keep routine
responses concise and reduce CPU time.

## Requirements

- Python 3.10 or newer
- A 64-bit operating system
- Enough free disk space for the GGUF model (about 5.03 GB)
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

For a standard installation, install dependencies and configure the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to change the defaults. The default
inference settings use four threads by default and a 4096-token context;
`N_CTX` can be adjusted when needed, and the app rejects a thread count other
than four. `MODEL_PATH` controls where the
GGUF file is stored, while `HOST` and `PORT` configure the Gradio listener.
Keep the default host (`127.0.0.1`) for local-only access.

## Run

```bash
python app.py
```

For Linux hosting or a Linux container, use the CPU-wheel startup script:

```bash
bash start.sh
```

`start.sh` installs a binary `llama-cpp-python` wheel from the official CPU
wheel index before installing the remaining requirements, then launches the
app. It deliberately omits `llama-cpp-python` from the second pip install so
pip cannot replace the CPU wheel with a PyPI build or fall back to compiling
from source. It disables pip's download cache to reduce disk use. The script
runs both pip installs every time it starts; pip leaves already-satisfied
packages installed. It launches the app through `cpu_limit.py`, which sets a
395% cgroup CPU quota when the host permits it, with a process-monitor fallback
for containers that do not expose a writable cgroup. Run `python app.py`
directly when you do not want the Linux CPU limiter.

The CPU wheel and its unpacked files, Gradio (whose wheel is about 64 MB),
other dependencies, and temporary files still require free container storage.
`--no-cache-dir` avoids retaining an additional pip download cache; it cannot
make an installation succeed when the container filesystem itself is full.
If Pterodactyl reports `No space left on device`, check storage and inodes with
`df -h` and `df -i` inside the container, then free or increase the server's
allocated storage before restarting. Allow more space than the largest
download because pip also needs room for unpacked packages and temporary files.

Open the local URL printed by Gradio (by default
`http://127.0.0.1:7860`). The model is downloaded when the first message is
sent, then kept loaded and reused for later turns. Closing the application
releases the model and its context. Use the sidebar to adjust the system prompt,
temperature, top-p, and maximum output length. The chat includes clear and
regenerate controls, and each response has a copy button. Conversations are
stored locally in `data/chats.sqlite3`, survive server restarts, and appear in
the history panel. Search finds chats by their opening prompt and recent user
messages. With cross-chat context enabled, the app adds short notes from up to
two earlier conversations whose topics overlap with the current prompt.

## Model and memory notes

The default Hugging Face repository is `Qwen/Qwen3-8B-GGUF`, with
`Qwen3-8B-Q4_K_M.gguf`. Its Q4_K_M quantization is about 5.03 GB. At the
configured 4096-token context, allow several additional GiB for the loaded
model and runtime. Hugging Face Hub uses its normal cache and
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
the loaded model, reserves the selected output limit (up to 3072 tokens)\nfrom the 4096-token window, and keeps
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
