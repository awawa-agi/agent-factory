# AgentFactory

**Language**: [English](README.md) | 日本語 | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

AgentFactory は LLM/MLLM のエージェント訓練に特化した強化学習フレームワークです。

--- 

## 開発動機
TRL は優秀ですが、transformers Trainer に過度に依存しており、改修が困難です。VeRL は強力そうですが、ray で管理されているため、コードが読みにくいです。そこで AgentFactory フレームワークを開発し、使いやすさ、シンプルさ、高速性、モジュール化された読みやすいコードを兼ね備えることを目指しました。
huggingface エコシステムをベースに開発し、accelerate を使用してマルチ GPU 訓練を管理しています。

## プロジェクトの特徴
- Async Rollout：マルチターンツール呼び出しの訓練速度を大幅に向上
- マルチモーダル対応：OpenAI O3 のような Thinking with images をサポート
- mcp をツールおよびインタラクティブ環境として使用することをサポート
- 高度にモジュール化された分離アーキテクチャ、訓練・rollout・vllm を異なるモジュールに分離し、開発・保守を容易化
- 主に huggingface システム（transformers, accelerate）上に構築、ray を使用せず、シンプルで理解しやすいコード
- 訓練では fsdp2 をサポート、推論では fp8, awq などの量子化高速化をサポート。liger-kernel による高速化と VRAM 使用量削減をサポート
- lora 訓練をサポート
- 呼び出し上限などの各種ツール制限をサポート

## 更新履歴

## 使用方法

### インストール

### データとツール

### クイックスタート

## ライセンス

## 謝辞
多くの内容は [TRL](https://github.com/huggingface/trl) から派生しています。以下の優秀なプロジェクト VeRL, [DeepEyes](https://github.com/Visual-Agent/DeepEyes) を参考にさせていただきました。

TODO:
- [ ] より効率的な loss のための logits_to_keep
- [ ] フルファインチューニングのための sglang
- [ ] ドキュメントの完善
- [ ] Save & Resume
- [ ] rollout failed filter
- [ ] rollout 開始時のファイル移動
- [ ] prefix, maybe without <think>
- [ ] Qwen2.5-VL abs/rel coordinate のテスト
- [ ] コード評価
- [ ] より多くの optimizer & scheduler のサポート