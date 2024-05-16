# CogVLM2

[Read this in English.](./README_en.md)

## 项目更新

- 🔥  **News**: ```2024/5/17``` 我们开源了 `cogvlm2-llama3-chat-19B` 和 `cogvlm2-llama3-chinese-chat-19B` 两款模型。

## 模型介绍

CogVLM 是一个强大的开源视觉语言模型（VLM）。我们推出了新一代的 CogVLM2 系列模型，相较于上一代CogVLM开源模型，CogVLM2系列开源模型的提升表现在：

1. 支持更长的文本长度。
2. 支持高达 1344 * 1344 的图片分辨率。
3. 加入了中文能力。

下表展现了 CogVLM2 视觉理解模型中的开源模型列表

| 模型名称   | cogvlm2-llama3-chat-19B             | cogvlm2-llama3-chinese-chat-19B     |
|--------|-------------------------------------|-------------------------------------|
| 基座模型   | Meta-Llama-3-8B-Instruct            | Meta-Llama-3-8B-Instruct            |
| 语言     | 英文                                  | 中文、英文                               |
| 模型大小   | 19B                                 | 19B                                 |
| 任务     | 图像理解，对话模型                           | 图像理解，对话模型                           |
| 模型链接   | [🤗 Huggingface]() [🤖ModelScope]() | [🤗 Huggingface]() [🤖ModelScope]() |
| Int4模型 | [🤗 Huggingface]() [🤖ModelScope]() | 暂未推出                                |
| 文本长度   | 8K                                  | 8K                                  |
| 图片分辨率  | 1344 * 1344                         | 1344 * 1344                         |

我们的开源模型相较于上一代CogVLM开源模型，在多项榜单中取得较好的成绩。其优异的表现能与部分的非开源模型进行同台竞技，如下表所示：

| Model                          | TextVQA  | DocVQA   | ChartQA | OCRbench | MMMU     | MMVet    | MMBench  |
|--------------------------------|----------|----------|---------|----------|----------|----------|----------|
| Mini-Gemini                    | 74.1     | -        | -       | -        | 48.0     | 59.3     | 80.6     |
| LLaVA-NeXT-LLaMA3              | -        | 78.2     | 69.5    | -        | 41.7     | -        | 72.1     |
| LLaVA-NeXT-110B                | -        | 85.7     | 79.7    | -        | 49.1     | -        | 80.5     |
| InternVL-1.5                   | 80.6     | 90.9     | **83.8**    | 720      | 46.8     | 55.4     | **82.3**     |
| QwenVL-Plus                    | 78.9     | 91.4     | 78.1    | 726      | 51.4     | 55.7     | 67.0     |
| Claude3-Opus                   | -        | 89.3     | 80.8    | 694      | **59.4** | 51.7     | 63.3     |
| Gemini Pro 1.5                 | 73.5     | 86.5     | 81.3    | -        | 58.5     | -        | -        |
| GPT-4V                         | 78.0     | 88.4     | 78.5    | 656      | 56.8     | **67.7** | 75.0     |
| CogVLM1.1 (Ours)               | 69.7     | -        | 68.3    | 590      | 37.3     | 52.0     | 65.8     |
| CogVLM2-LLaMA3 (Ours)          | 84.2     | **92.3** | 81.0    | 756      | 44.3     | 60.4     | 80.5 |
| CogVLM2-LLaMA3-Chinese  (Ours) | **85.0** | 88.4     | 74.7    | **780**  | 42.8     | 60.5     | 78.9     |

所有评测都是在不使用任何外部OCR工具(“only pixel”)的情况下获得的。

## 项目结构

本开源仓库将带领开发者快速上手 CogVLM2 开源模型的基础调用方式、微调示例、OpenAI API格式调用示例等。具体项目结构如下，您可以点击进入对应的教程链接：

+ [basic_demo](basic_demo/README.md) - 基础调用方式,包含了 CLI, WebUI 和 OpenAI API等模型推理调用方式。如果您是第一次使用 CogVLM2 开源模型，建议您从这里开始。
+ [finetune_demo](finetune_demo/README.md) - 微调示例，包含了微调语言模型的示例。

## 提出支持和与我们联系

+ 加入我们的 [微信](resources/WECHAT.md) 群，与其他 CogVLM的小伙伴一起沟通。
+ 如果你想作出贡献，请严格按照 [Issue规范](.github/ISSUE_TEMPLATE) 和 [PR规范](.github/PULL_REQUEST_TEMPLATE) 作出贡献。 



