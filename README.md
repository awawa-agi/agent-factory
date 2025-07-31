# AgentFactory

**Language**: English | [日本語](README_jp.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

AgentFactory is a reinforcement learning framework focused on LLM/MLLM agent training.

--- 

## Development Motivation
TRL is excellent but overly dependent on transformers Trainer, making modifications difficult. VeRL seems powerful but uses ray for management, making the code hard to read. Therefore, we developed the AgentFactory framework, aiming to combine usability, simplicity, speed, and modular readable code.
Built on the huggingface ecosystem, using accelerate to manage multi-GPU training.

## Project Features
- Async Rollout, significantly improving training speed for multi-turn tool calling
- Multimodal support: supports OpenAI O3-like Thinking with images
- Support for using mcp as tools and interactive environment
- Highly modular decoupled architecture with training, rollout, and vllm as separate modules for easy development and maintenance
- Primarily built on huggingface ecosystem (transformers, accelerate), no ray dependency, simple and readable code
- Training supports fsdp2, inference supports fp8, awq and other quantization acceleration. Supports liger-kernel for acceleration and VRAM reduction
- Support for lora training
- Support for various tool limitations, such as call limits, etc.

## Changelog

## How to Use

### Installation

### Data and Tools

### Quick Start

## License

## Acknowledgments
Much content is derived from [TRL](https://github.com/huggingface/trl). Thanks to the following excellent projects VeRL, [DeepEyes](https://github.com/Visual-Agent/DeepEyes) for reference.

TODO:
- [ ] logits_to_keep for more efficient loss
- [ ] sglang for full finetuning
- [ ] Complete documentation
- [ ] Save & Resume
- [ ] rollout failed filter
- [ ] file moving when start rollout
- [ ] prefix, maybe without <think>
- [ ] Test Qwen2.5-VL abs/rel coordinate
- [ ] Evaluate code
- [ ] Support more optimizer & scheduler
- [ ] torch.compile & FP8 training & QLoRA