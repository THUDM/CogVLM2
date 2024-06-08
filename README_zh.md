# CogVLM2

[Read this in English.](./README_en.md)


<div align="center">
<img src=resources/logo.svg width="40%"/>
</div>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信</a> · 💡 立刻 <a href="http://36.103.203.44:7861/" target="_blank">在线体验</a>
</p>
<p align="center">
📍在 <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">开放平台</a> 体验更大规模的 CogVLM 模型。
</p>


## 近期更新
- 🔥🔥 **News**：``2024/6/8``：我们发布 [CogVLM2 TGI 模型权重](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-tgi)，这是一个可以在 [TGI](https://huggingface.co/docs/text-generation-inference/en/index) 环境加速推理的模型。
- 🔥 **News**：``2024/6/5``：我们发布 [GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b)，它使用与 CogVLM2 相同的数据和训练配方，但以 GLM-9B 作为语言主干。我们删除了视觉专家，以将模型大小减小到 13B。更多详细信息，请参阅 [GLM-4 repo](https://github.com/THUDM/GLM-4/)。
- 🔥 **News**：``2024/5/24``：我们发布了 Int4 版本模型，仅需要 16GB 显存即可进行推理。欢迎前来体验！
- 🔥 **News**：``2024/5/20``：我们发布了下一代模型 CogVLM2，它基于 llama3-8b，在大多数情况下与 GPT-4V 相当（或更好）！欢迎下载！

## 模型介绍

我们推出了新一代的 **CogVLM2**
系列模型并开源了两款基于 [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
开源模型。与上一代的 CogVLM 开源模型相比，**CogVLM2** 系列开源模型具有以下改进：

1. 在许多关键指标上有了显著提升，例如 `TextVQA`, `DocVQA`。
2. 支持 **8K** 文本长度。
3. 支持高达 **1344 * 1344** 的图像分辨率。
4. 提供支持**中英文双语**的开源模型版本。

您可以在下表中看到 **CogVLM2** 系列开源模型的详细信息：

| 模型名称    | cogvlm2-llama3-chat-19B                                                                                                                                                                                                               | cogvlm2-llama3-chinese-chat-19B                                                                                                                                                                                                                              |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 基座模型    | Meta-Llama-3-8B-Instruct                                                                                                                                                                                                              | Meta-Llama-3-8B-Instruct                                                                                                                                                                                                                                     |
| 语言      | 英文                                                                                                                                                                                                                                    | 中文、英文                                                                                                                                                                                                                                                        |
| 模型大小    | 19B                                                                                                                                                                                                                                   | 19B                                                                                                                                                                                                                                                          |
| 任务      | 图像理解，对话模型                                                                                                                                                                                                                             | 图像理解，对话模型                                                                                                                                                                                                                                                    |
| 模型链接    | [🤗 Huggingface](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B)  [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chat-19B/)  [💫 Wise Model](https://wisemodel.cn/models/ZhipuAI/cogvlm2-llama3-chat-19B/) | [🤗 Huggingface](https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B) [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B/)  [💫 Wise Model](https://wisemodel.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B/) |
| 体验链接    | [📙 Official Page](http://36.103.203.44:7861/)                                                                                                                                                                                        | [📙 Official Page](http://36.103.203.44:7861/) [🤖 ModelScope](https://modelscope.cn/studios/ZhipuAI/Cogvlm2-llama3-chinese-chat-Demo/summary)                                                                                                               |
| Int4 模型 | [🤗 Huggingface](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4)  [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chat-19B-int4)                                                                       | [🤗 Huggingface](https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B-int4) [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B-int4)                                                                               |
| 文本长度    | 8K                                                                                                                                                                                                                                    | 8K                                                                                                                                                                                                                                                           |
| 图片分辨率   | 1344 * 1344                                                                                                                                                                                                                           | 1344 * 1344                                                                                                                                                                                                                                                  |

## Benchmark

我们的开源模型相较于上一代 CogVLM 开源模型，在多项榜单中取得较好的成绩。其优异的表现能与部分的非开源模型进行同台竞技，如下表所示：

| Model                          | Open Source | LLM Size | TextVQA  | DocVQA   | ChartQA  | OCRbench | MMMU     | MMVet    | MMBench  |
|--------------------------------|-------------|----------|----------|----------|----------|----------|----------|----------|----------|
| CogVLM1.1                      | ✅           | 7B       | 69.7     | -        | 68.3     | 590      | 37.3     | 52.0     | 65.8     |
| LLaVA-1.5                      | ✅           | 13B      | 61.3     | -        | -        | 337      | 37.0     | 35.4     | 67.7     |
| Mini-Gemini                    | ✅           | 34B      | 74.1     | -        | -        | -        | 48.0     | 59.3     | 80.6     |
| LLaVA-NeXT-LLaMA3              | ✅           | 8B       | -        | 78.2     | 69.5     | -        | 41.7     | -        | 72.1     |
| LLaVA-NeXT-110B                | ✅           | 110B     | -        | 85.7     | 79.7     | -        | 49.1     | -        | 80.5     |
| InternVL-1.5                   | ✅           | 20B      | 80.6     | 90.9     | **83.8** | 720      | 46.8     | 55.4     | **82.3** |
| QwenVL-Plus                    | ❌           | -        | 78.9     | 91.4     | 78.1     | 726      | 51.4     | 55.7     | 67.0     |
| Claude3-Opus                   | ❌           | -        | -        | 89.3     | 80.8     | 694      | **59.4** | 51.7     | 63.3     |
| Gemini Pro 1.5                 | ❌           | -        | 73.5     | 86.5     | 81.3     | -        | 58.5     | -        | -        |
| GPT-4V                         | ❌           | -        | 78.0     | 88.4     | 78.5     | 656      | 56.8     | **67.7** | 75.0     |
| CogVLM2-LLaMA3 (Ours)          | ✅           | 8B       | 84.2     | **92.3** | 81.0     | 756      | 44.3     | 60.4     | 80.5     |
| CogVLM2-LLaMA3-Chinese  (Ours) | ✅           | 8B       | **85.0** | 88.4     | 74.7     | **780**  | 42.8     | 60.5     | 78.9     |

所有评测都是在不使用任何外部OCR工具(“only pixel”)的情况下获得的。

## 项目结构

本开源仓库将带领开发者快速上手 **CogVLM2** 开源模型的基础调用方式、微调示例、OpenAI API格式调用示例等。具体项目结构如下，您可以点击进入对应的教程链接：

## [basic_demo](basic_demo/README.md) 文件夹包括：
  + **CLI** 演示。
  + **CLI** 演示，使用多个GPU。
  + **Web** 演示，由 chainlit 提供。
  + **API** 服务器，采用 OpenAI 格式。
  + **Int4** 可以通过 `--quant 4` 轻松启用，内存使用为16GB。

## [finetune_demo](finetune_demo/README.md) 文件夹包括：
  + [**peft**](https://github.com/huggingface/peft) 框架的高效微调示例。
  + [TODO] 使用 [**sat**](http://github.com/THUDM/SwissArmyTransformer) 框架的可靠微调示例。
  + [TODO] 将 `sat` 格式的检查点转换为 `huggingface` 格式的转换脚本。

## 友情链接

除了官方提供的推理代码，还有以下由社区提供的推理方案可以参考。包括: 
 + [**xinference**](https://github.com/xorbitsai/inference/pull/1551)

## 模型协议

该模型根据 [CogVLM2 LICENSE](MODEL_LICENSE) 许可证发布。对于使用了Meta Llama
3基座模型构建的模型，需要同时遵守 [LLAMA3_LICENSE](https://llama.meta.com/llama3/license/) 许可证。

## 引用

如果您发现我们的工作有所帮助，请考虑引用以下论文:

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