# AgentFactory

**Language**: [English](README.md) | [日本語](README_jp.md) | [简体中文](README_zh-CN.md) | 繁體中文

AgentFactory 是一個專注於 LLM/MLLM 的 agent 訓練的強化學習框架。

--- 

## 開發初衷
TRL 很好用，但過於依賴 transformers Trainer，不好進行改動。 VeRL 好像很強大，但使用 ray 管理，程式好難讀。所以開發了 AgentFactory 框架，希望可以兼具好用、間單、快速、程式模組化好讀。
基於 huggingface 生態系開發，使用 accelerate 管理多 gpus 訓練。

## 项目特色
- Async Rollout，大幅提昇multi-turn工具呼叫的訓練速度
- 多模態支援：支援類似 OpenAI O3 的 Thinking with images
- 支援使用 mcp 作為工具與交互環境
- 高度模組化的分離式架構，訓練、rollout、vllm 分為不同模組，方便開發與維護
- 主要建立於 huggingface 系統上 (transformers, accelerate)，沒有使用 ray，程式簡單好讀懂
- 訓練支援 fsdp2，推理支援 fp8, awq 等量化加速。支援 liger-kernel 加速並減少 VRAM 開銷。
- 支援 lora 訓練
- 支援多種工具限制，如呼叫上限等等

## 更新日志

## 如何使用

### 安裝

#### 系統需求
- Python ≥3.11
- CUDA ≥11.8 (推薦 12.1+)

#### 兩階段安裝
由於 flash-attn 的 PyTorch 依賴問題，需要分兩階段安裝：

```bash
# 階段一：先安裝 PyTorch 和其他依賴
uv sync --no-install-package flash-attn

# 階段二：安裝 flash-attn
uv sync

# 開發環境（可選）
uv sync --dev
```

### 數據與工具

### 快速开始

## 協議

## 致謝
許多內容來自 [TRL](https://github.com/huggingface/trl) 。感謝以下優秀的項目 [verl-agent](https://github.com/langfengQ/verl-agent), [VeRL](https://github.com/volcengine/verl), [DeepEyes](https://github.com/Visual-Agent/DeepEyes) 作為參考。

TODO:
- [ ] logits_to_keep for more efficient loss
- [ ] sglang for full finetuning
- [ ] 完善文檔
- [ ] Save & Resume
- [ ] rollout failed filter
- [ ] file moving when start rollout
- [ ] prefix, maybe without <think>
- [ ] Test Qwen2.5-VL abs/rel coordinate
- [ ] Evaluate code
- [ ] Support more optimizer & scheduler