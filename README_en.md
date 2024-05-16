# CogVLM2

## Project updates

- 🔥 **News**: ```2024/5/17``` We open sourced two models `cogvlm2-llama3-chat-19B`
  and `cogvlm2-llama3-chinese-chat-19B`.

## Model introduction

CogVLM is a powerful open source visual language model (VLM). We have launched a new generation of CogVLM2 series
models. Compared with the previous generation of CogVLM open source models, the improvements of the CogVLM2 series of
open source models are as follows:

1. Support longer text length.
2. Support image resolution up to 1344 * 1344.
3. Added Chinese language capability.

The following table shows the list of open source models in the CogVLM2 visual understanding model

| Model name       | cogvlm2-llama3-chat-19B             | cogvlm2-llama3-chinese-chat-19B     |
|------------------|-------------------------------------|-------------------------------------|
| Base Model       | Meta-Llama-3-8B-Instruct            | Meta-Llama-3-8B-Instruct            |
| Language         | English                             | Chinese, English                    |
| Model size       | 19B                                 | 19B                                 |
| Task             | Image understanding, dialogue model | Image understanding, dialogue model |
| Model link       | [🤗 Huggingface]() [🤖ModelScope]() | [🤗 Huggingface]() [🤖ModelScope]() |
| Int4 model       | [🤗 Huggingface]() [🤖ModelScope]() | Not yet launched                    |
| Text length      | 8K                                  | 8K                                  |
| Image resolution | 1344 * 1344                         | 1344 * 1344                         |

Compared with the previous generation CogVLM open source model, our open source model has achieved better results in
many lists. Its excellent performance can compete with some non-open source models, as shown in the following table:

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

All reviews were obtained without using any external OCR tools ("pixel only").

## Project structure

This open source repos will help developers to quickly get started with the basic calling methods of the CogVLM2 open
source model, fine-tuning examples, OpenAI API format calling examples, etc. The specific project structure is as
follows, you can click to enter the corresponding tutorial link:

+ [basic_demo](basic_demo/README_en.md) -Basic calling methods include model inference calling methods such as CLI, WebUI and OpenAI API. If you are using the CogVLM2 open source model for the first time, we recommend you start here.
+ [finetune_demo](finetune_demo/README_en.md) - Fine-tuning examples, including examples of fine-tuning language models.

## Raise support and contact us

+ Join our [WeChat](resources/WECHAT.md) group to communicate with other CogVLM partners.
+ If you want to contribute, please strictly follow the [Issue specification](.github/ISSUE_TEMPLATE)
  and [PR specification](.github/PULL_REQUEST_TEMPLATE) to contribute.
