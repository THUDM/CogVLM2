# CogVLM2

[中文版README](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="20%"/>
</div>
<p align="center">
🤗 <a href="https://huggingface.co/THUDM/CogVLM2" target="_blank">HF Repo</a> • 🤖 <a href="https://modelscope.cn/models/ZhipuAI/CogVLM2" target="_blank">魔搭社区</a>
</p>
<p align="center">
    👋 Join our <a href="resources/WECHAT.md" target="_blank">Wechat</a>
</p>
<p align="center">
📍Experience the larger-scale CogVLM model on the <a href="https://open.bigmodel.cn/dev/api#super-humanoid">ZhipuAI Open Platform</a>.
</p>

We launch a new generation of **CogVLM2** series of models and open source two models based on [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Compared with the previous generation of CogVLM open source models, the CogVLM2 series of open source models have the following improvements:

1. Significant improvements in many benchmarks such as `TextVQA`, `DocVQA`.
2. Support **8K** content length.
3. Support image resolution up to **1344 * 1344**.
4. Provide an open source model version that supports both **Chinese and English**.

You can see the details of the **CogVLM2** family of open source models in the table below:

| Model name       | cogvlm2-llama3-chat-19B             | cogvlm2-llama3-chinese-chat-19B     |
|------------------|-------------------------------------|-------------------------------------|
| Base Model       | Meta-Llama-3-8B-Instruct            | Meta-Llama-3-8B-Instruct            |
| Language         | English                             | Chinese, English                    |
| Model size       | 19B                                 | 19B                                 |
| Task             | Image understanding, dialogue model | Image understanding, dialogue model |
| Model link       | [🤗 Huggingface]() [🤖ModelScope]() | [🤗 Huggingface]() [🤖ModelScope]() |
| Int4 model       | Not yet launched    | Not yet launched                    |
| Text length      | 8K                                  | 8K                                  |
| Image resolution | 1344 * 1344                         | 1344 * 1344                         |

## Benchmark

Our open source models have achieved good results in many lists compared to the previous generation of CogVLM open source models. Its excellent performance can compete with some non-open source models, as shown in the table below:

| Model                          | Open Source | LLM Size | TextVQA  | DocVQA   | ChartQA  | OCRbench | MMMU     | MMVet    | MMBench  |
|--------------------------------|-------------|----------|----------|----------|----------|----------|----------|----------|----------|
| CogVLM1.1            | ✅           | 7B       | 69.7     | -        | 68.3     | 590      | 37.3     | 52.0     | 65.8     |
| LLaVA-1.5                      | ✅           | 13B      | 61.3     | -        | -        | 337      | 37.0     | 35.4     | 67.7     |
| Mini-Gemini                    | ✅           | 34B      | 74.1     | -        | -        | -        | 48.0     | 59.3     | 80.6     |
| LLaVA-NeXT-LLaMA3              | ✅           | 8B       | -        | 78.2     | 69.5     | -        | 41.7     | -        | 72.1     |
| LLaVA-NeXT-110B                | ✅           | 110B     | -        | 85.7     | 79.7     | -        | 49.1     | -        | 80.5     |
| InternVL-1.5                   | ✅           | 20B      | 80.6     | 90.9     | **83.8** | 720      | 46.8     | 55.4     | **82.3** |
| QwenVL-Plus                    | ❌           | -        | 78.9     | 91.4     | 78.1     | 726      | 51.4     | 55.7     | 67.0     |
| Claude3-Opus                   | ❌           | -        | -        | 89.3     | 80.8     | 694      | **59.4** | 51.7     | 63.3     |
| Gemini Pro 1.5                 | ❌           | -        | 73.5     | 86.5     | 81.3     | -        | 58.5     | -        | -        |
| GPT-4V                         | ❌           | -        | 78.0     | 88.4     | 78.5     | 656      | 56.8     | **67.7** | 75.0     |
| **CogVLM2-LLaMA3**         | ✅           | 8B       | 84.2     | **92.3** | 81.0     | 756      | 44.3     | 60.4     | 80.5     |
| **CogVLM2-LLaMA3-Chinese** | ✅           | 8B       | **85.0** | 88.4     | 74.7     | **780**  | 42.8     | 60.5     | 78.9     |

All reviews were obtained without using any external OCR tools ("pixel only").

## Project structure

This open source repos will help developers to quickly get started with the basic calling methods of the CogVLM2 open
source model, fine-tuning examples, OpenAI API format calling examples, etc. The specific project structure is as
follows, you can click to enter the corresponding tutorial link:

+ [basic_demo](basic_demo/README_en.md) - Basic calling methods include model inference calling methods such as CLI, WebUI and OpenAI API. If you are using the CogVLM2 open source model for the first time, we recommend you start here.
+ [finetune_demo](finetune_demo/README_en.md) - Fine-tuning examples, including examples of fine-tuning language models.

## License

This model is released under the CogVLM2 [LICENSE](LICENSE.md). For models built with Meta Llama 3, please also adhere to the [LLAMA3_LICENSE](LLAMA3_LICENSE.md).

## Citation

If you find our work helpful, please consider citing the following papers

```
@misc{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
