# AgentFactory

**Language**: [English](README.md) | [日本語](README_jp.md) | 简体中文 | [繁體中文](README_zh-TW.md)

AgentFactory 是一个专注于 LLM/MLLM 的 agent 训练的强化学习框架。

--- 

## 开发初衷
TRL 很好用，但过于依赖 transformers Trainer，不好进行改动。 VeRL 好像很强大，但使用 ray 管理，程序好难读。所以开发了 AgentFactory 框架，希望可以兼具好用、简单、快速、程序模块化好读。
基于 huggingface 生态系开发，使用 accelerate 管理多 gpus 训练。

## 项目特色
- Async Rollout，大幅提升multi-turn工具调用的训练速度
- 多模态支持：支持类似 OpenAI O3 的 Thinking with images
- 支持使用 mcp 作为工具与交互环境
- 高度模块化的分离式架构，训练、rollout、vllm 分为不同模块，方便开发与维护
- 主要建立于 huggingface 系统上 (transformers, accelerate)，没有使用 ray，程序简单好读懂
- 训练支持 fsdp2，推理支持 fp8, awq 等量化加速。支持 liger-kernel 加速并减少 VRAM 开销。
- 支持 lora 训练
- 支持多种工具限制，如调用上限等等

## 更新日志

## 如何使用

### 安装

### 数据与工具

### 快速开始

## 协议

## 致谢
许多内容来自 [TRL](https://github.com/huggingface/trl) 。感谢以下优秀的项目 [verl-agent](https://github.com/langfengQ/verl-agent), [VeRL](https://github.com/volcengine/verl), [DeepEyes](https://github.com/Visual-Agent/DeepEyes) 作为参考。

TODO:
- [ ] logits_to_keep for more efficient loss
- [ ] sglang for full finetuning
- [ ] 完善文档
- [ ] Save & Resume
- [ ] rollout failed filter
- [ ] file moving when start rollout
- [ ] prefix, maybe without <think>
- [ ] Test Qwen2.5-VL abs/rel coordinate
- [ ] Evaluate code
- [ ] Support more optimizer & scheduler
- [ ] torch.compile & FP8 training & QLoRA