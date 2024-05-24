# Fine-tune the CogVLM2 model

[中文版README](./README_zh.md)

## Note

+ This code only provides fine-tuning examples for the huggingface version model 'cogvlm2-llama3-chat-19B'.
+ Only examples of fine-tuning language models are provided.
+ Only provide Lora fine-tuning examples.
+ Only provide examples of fine-tuning the dialogue model.
+ We currently do not support using 'zero3' fine-tuning, which may result in the model not being able to read.

## Minimum configuration

- We Only test A100 GPUs with 80GB memory for finetune. It requires at least 73GB of GPU memory using 8 GPUs with zero2.
- Tensor parallelism is not supported yet, that is, the model is split into multiple graphics cards for fine-tuning.

## Start fine-tuning

1. Download the data set and install dependencies

In this demo, developers can use the [CogVLM-SFT-311K](https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K) open
source data set provided by us or build their own data set in the same format for fine-tuning. .

The data format is as follows:

+ The data set consists of two folders, `images` and `labels` (in CogVLM-SFT-311K, they are `labels_en` and `labels_zh`,
  corresponding to Chinese and English labels respectively.
  In the fine-tuning code, you can modify these two lines of code to modify the folder name.

```python
self.image_dir = os.path.join(root_dir, 'images')
self.label_dir = os.path.join(root_dir, 'labels_en')  # or 'labels_zh' or 'labels' can be modified by yourself
```

+ Image files are stored in the `images` folder, and corresponding label files are stored in the `labels` folder. There
  is a one-to-one correspondence between the names of pictures and label files. The format of image files is `jpg`, and
  the format of label files is `json`.
+ Each tag file contains a dialogue. The dialogue consists of two roles: `user` and `assistant`. The dialogue content of
  each role consists of two fields: `role` and `content`. As shown in the fields below.

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

Before starting fine-tuning, you need to install the relevant dependencies. you also need to install the dependencies in the  [basic_demo](../basic_demo/requirements.txt).

```bash
pip install -r requirements.txt
```

**Note**: `mpi4py` may need to install other Linux dependency packages. Please install it yourself according to your
system environment.

2. Run the fine-tuning program

We provide a fine-tuning script `peft_lora.py` that uses multiple cards on a single machine (including a single card).
You can start fine-tuning by running the following command.

```bash
deepspeed peft_lora.py --ds_config ds_config.yaml
```

The figure below shows the memory usage during fine-tuning.

Parameter information:

+ `max_input_len`: 512
+ `max_output_len`: 512
+ `batch_size_per_gpus`: 1
+ `lora_target`: vision_expert_query_key_value

GPU memory usage:

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

While the code is running, Loss data will be recorded by tensorboard to facilitate visual viewing of Loss convergence.

```shell
tensorboard --logdir=output
```

**Note**: We strongly recommend that you use the `BF16` format for fine-tuning to avoid the problem of Loss being `NaN`.

3. Inference on the fine-tuned model

By running `peft_infer.py` you can use the fine-tuned model to generate text. You need to configure the fine-tuned model
address according to the configuration requirements in the code. Then run:

```shell
python peft_infer.py
```

You can use the fine-tuned model for inference.