# 微调 CogVLM2 模型

[Read this in English.](./README.md)

运行本demo来使用Lora微调 CogVLM2 中的**语言模型**部分。

## 注意

+ 本代码仅提供了 huggingface 版本模型 `cogvlm2-llama3-chat-19B` 的微调示例。
+ 仅提供了微调语言模型的示例。
+ 仅提供Lora微调示例。
+ 仅提供对话模型微调示例。
+ 暂不支持使用 `zero3` 微调，这可能出现 模型无法读取的情况。

## 最低配置

- 我们仅在具有80GB内存的A100 GPU上进行了微调测试。使用零冗余优化策略2（zero2）时，至少需要73GB的GPU内存，并且需要8个GPU。
- 暂不支持 Tensor 并行，即模型拆分到多张显卡微调。

## 开始微调

1. 下载数据集和安装依赖

本demo中，开发者可以使用由我们提供[CogVLM-SFT-311K](https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K)
开源数据集或自行构建相同格式的数据集进行微调。

数据格式如下:

+ 数据集由 `images` 和 `labels` 两个文件夹组成 （在 CogVLM-SFT-311K 中 为 `labels_en` 和 `labels_zh`，分别对应中英文标签。
  在微调代码中，你可以修改这两行代码来修改文件夹名称。

```python
self.image_dir = os.path.join(root_dir, 'images')
self.label_dir = os.path.join(root_dir, 'labels_en')  # or 'labels_zh' or 'labels' 可以自行修改
```

+ `images` 文件夹中存放了图片文件，`labels`
  文件夹中存放了对应的标签文件。图片和标签文件的名称一一对应。图片文件的格式为 `jpg`，标签文件的格式为 `json`。
+ 每个标签文件中包含了一段对话，对话由 `user` 和 `assistant` 两个角色组成，每个角色的对话内容由 `role` 和 `content`
  两个字段组成。如下字段所示。

```
{
  "conversations": [
    {
      "role": "user",
      "content": "What can be inferred about the zebras' behavior and surroundings?"
    },
    {
      "role": "assistant",
      "content": "Based on the image, we can infer that the two zebras are likely seeking relief from the sun's heat, as they are standing side by side under the branches of a thorny tree. This shade-providing tree offers some respite from the sun, possibly during the hottest part of the day. The zebras are in a green field with grass, providing them with an ideal environment to graze and eat while staying near their source of shelter. This shows that the zebras' behavior is influenced by the conditions and available resources in their surroundings. It also highlights that these animals adopt strategies to adapt to the fluctuating conditions of their environment, such as cooperation and seeking shelter, to survive and thrive in their natural habitat."
    }
  ]
}
```

在开始微调之前，需要安装相关的依赖。请注意，你还需要安装好 [basic_demo](../basic_demo/requirements.txt) 中的依赖。

```bash
pip install -r requirements.txt
```

**注意**: `mpi4py` 可能需要安装别的 Linux 依赖包。请根据您的系统环境自行安装。

2. 运行微调程序

我们提供了使用单机多卡（包含单卡）的微调脚本 `peft_lora.py`。您可以通过运行以下命令来启动微调。

```bash
deepspeed peft_lora.py --ds_config ds_config.yaml
```

下图展现了微调过程中的显存占用情况

参数信息：

+ `max_input_len`: 512
+ `max_output_len`: 512
+ `batch_size_per_gpus`: 1
+ `lora_target`: vision_expert_query_key_value

显存占用情况：

```shell
+-------------------------------------------------------------+
| Processes:                                                  |
|  GPU   GI   CI        PID   Type   Process name  GPU Memory |
|        ID   ID                                      Usage   |
|=============================================================|
|    0   N/A  N/A    704914      C   python          72442MiB |
|    1   N/A  N/A    704915      C   python          72538MiB |
|    2   N/A  N/A    704916      C   python          72538MiB |
|    3   N/A  N/A    704917      C   python          72538MiB |
|    4   N/A  N/A    704918      C   python          72538MiB |
|    5   N/A  N/A    704919      C   python          72538MiB |
|    6   N/A  N/A    704920      C   python          72538MiB |
|    7   N/A  N/A    704921      C   python          72442MiB |
+-------------------------------------------------------------+
```

在代码运行中，Loss数据会被 tensorboard记录，方便可视化查看Loss收敛情况。

```shell
tensorboard --logdir=output
```

**注意**: 我们强烈推荐您使用 `BF16` 格式进行微调，以避免出现 Loss 为 `NaN`的问题。

3. 推理微调后的模型

运行 `peft_infer.py`，你可以使用微调后的模型生成文本。您需要按照代码中的配置要求，配置微调后的模型地址。然后运行:

```shell
python peft_infer.py
```

即可使用微调的模型进行推理。

