# NVIDIA NIM Agents

This directory provides an extension for `pi` to interact with the NVIDIA NIM (NVIDIA Inference Microservices) platform. It allows the agent to leverage a vast array of high-performance models, including state-of-the-art reasoning, coding and vision models.

## Available Model Categories

The extension automatically categorizes models from the NVIDIA NIM API:

- **Reasoning/Thinking Models**: Specialized models (e.g., DeepSeek-V4, Qwen3-Thinking, Nemotron-Reasoning) that use `chat_template_kwargs` to enable internal chain-of-thought processing.
- **Vision Models**: Models capable of processing image inputs (e.g., Llama-3.2-Vision, Phi-4-Multimodal).
- **Coding Models**: High-performance models optimized for software engineering (e.g., Qwen3-Coder).

## Configuration & Setup

### Environment Variables
To use these agents, you must provide an NVIDIA API key:
- `NVIDIA_NIM_API_KEY`: Primary key environment variable.
- `NVIDIA_API_KEY`: Secondary fallback variable.

### Installation
Load the extension using:
```bash
pi -e ./path/to/pi-nvidia-nim
```

## Usage

Once the extension is loaded, you can switch models using the `/model` command in `pi`. Search for `nvidia-nim/` to see all available models provided by this extension.

### Enabling Thinking/Reasoning
For models listed in `THINKING_CONFIGS` (like DeepSeek or Qwen3), you can enable reasoning through the `pi` interface. The extension automatically maps `pi`'s reasoning levels to the specific `chat_template_kwargs` required by the NIM API.

## Model Discovery
The extension performs dynamic discovery on `session_start`. It first registers a curated list of "featured" models and then queries the NVIDIA NIM `/models` endpoint to add any new available models to the registry.
