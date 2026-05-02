---
title: "基于 Transformers 的 LLM 训练流程实践：从预训练、SFT 到 LoRA"
description: "按流程梳理 Happy-LLM 第六章的训练实践主线，包括模型初始化、预训练数据处理、Trainer/DeepSpeed、SFT 标签构造与 LoRA/peft 微调"
date: 2026-04-21
slug: happy-llm-transformers-training-practice
image: cover.svg
categories:
    - LLM-Learning
tags:
    - LLM
    - Happy-LLM
    - Transformers
    - 预训练
    - SFT
    - LoRA
    - DeepSpeed
math: true
---

第六章的重点很贴近主流工程实践：用 `Transformers + datasets + Trainer + DeepSpeed + peft` 这套生态把 LLM 训练流程串起来。

如果只想先抓住这一章真正要记住的东西，我会把它压成一句话：

> **模型怎么初始化、数据怎么整理进 Trainer、预训练和 SFT 的标签怎么分开、资源不够时怎么换成 LoRA，这四件事就是第六章的主线。**

这篇文章会顺着训练流程，把第六章最值得回查的部分整理成一篇流程总结文。

---

## 一、先把第六章的训练流程压成一张图

如果从工程视角看，第六章其实讲的是这样一条链路：

```text
模型配置与 tokenizer
    -> 预训练数据处理
    -> TrainingArguments + Trainer
    -> DeepSpeed 分布式启动
    -> SFT 数据构造
    -> LoRA / peft 高效微调
```

这条链路里最容易混淆的一点，是很多人会把注意力放在 `Trainer` 或者 `DeepSpeed` 上，觉得“框架”本身最重要。  
但顺着这一章往下看，会更清楚地发现：

- `Transformers` 负责把模型和训练流程抽象成统一接口
- `datasets` 负责把原始文本或对话数据整理成模型可训练的张量格式
- `Trainer` 负责承接训练参数、数据集、保存和日志等通用逻辑
- `DeepSpeed` 负责把这件事推到多卡和更大规模上
- `peft` 负责在资源受限时把全量微调切成 LoRA 这种高效微调路径

换句话说，第六章真正搭起来的是一整条**框架化训练骨架**。

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-2.png" alt="Hugging Face Transformers 模型社区" width="90%" />
  <p>图：第六章借助 Hugging Face 生态把模型、数据集与训练框架串到了一起</p>
</div>

---

## 二、第一步：先把模型和 tokenizer 初始化起来

这一步在整个流程里负责的是：**把“要训练什么模型”这件事明确下来。**

第六章选的是 `Qwen-2.5-1.5B` 这条线。这里最值得记住的是 `Transformers` 里初始化模型的两种常见方式：

1. 基于配置从零初始化一个模型
2. 基于已有权重加载一个预训练模型

对应到代码里，主干大概就是下面这样：

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
```

如果要直接继承已有权重，通常会改成：

```python
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
```

这里我觉得最重要的记忆点有两个。

### 2.1 `Auto*` 这套接口，本质上是在替你接住“模型族差异”

只要配置和权重是兼容的，`AutoConfig`、`AutoModelForCausalLM`、`AutoTokenizer` 就能把很多底层结构细节统一掉。  
这也是为什么第六章把模型训练的重心转向数据、参数和流程。

### 2.2 训练阶段不同，模型加载方式可能不同，但 tokenizer 契约仍然很关键

无论是 Pretrain 还是 SFT，tokenizer 都是一套输入协议。它定义的是：

- 词表怎么映射
- 特殊 token 怎么表示
- 对话边界怎么标记
- padding 和 EOS 怎么约定

尤其到了后面的 SFT，这个 tokenizer 契约会直接决定 `chat template` 和 `labels` 的构造方式。

这一部分真正该记住的点是：**第六章里的模型初始化负责把后面所有训练步骤接到统一的 Hugging Face 模型接口上。**

---

## 三、第二步：把预训练数据整理成 Trainer 能直接吃的样子

这一步负责的是：**把原始文本数据变成 CLM 训练样本。**

第六章的预训练流程仍然是最经典的 Causal Language Modeling，也就是预测下一个 token。但进入 `datasets` 生态后，数据预处理的写法会更像一条流水线：

1. 用 `load_dataset` 读入原始 JSON/JSONL
2. 用 tokenizer 把文本转成 token IDs
3. 把多个文本段拼起来，切成固定长度 block
4. 令 `labels = input_ids.copy()`

最核心的代码骨架是这段：

```python
from datasets import load_dataset
from itertools import chain

ds = load_dataset("json", data_files=train_file)

def tokenize_function(examples):
    return tokenizer([item for item in examples["text"]])

tokenized_datasets = ds.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
)

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
```

### 3.1 `labels = input_ids.copy()` 是预训练阶段最关键的记忆锚点

这其实是在明确：**Pretrain 阶段的监督目标就是整段文本自身。**

也就是说，预训练数据的重点不在“问题-答案”格式，而在：

- 先把文本稳定 tokenize
- 再拼接成足够长的训练块
- 让每个块都作为一个 CLM 样本参与训练

这里为什么要做 `group_texts`，也很值得记一下。因为预训练时通常更关心吞吐和效率，而不是保持每段原始文本的天然边界。把多个片段拼成固定长度 block，可以显著减少 padding 浪费，让显卡更多时间在算有效 token。

### 3.2 预训练数据处理的核心是整理训练协议

一旦进入框架训练，真正重要的问题就变成了：

- block 多长
- 是否拼接多个样本
- `labels` 怎样和 `input_ids` 对齐
- `attention_mask` 由谁生成

这些细节表面上是数据预处理，实际上已经在决定模型训练看到的输入分布。

这一部分真正该记住的点是：**在第六章里，预训练数据处理把“文本 -> token -> block -> labels”的流水线标准化到 `datasets.map` 里。**

---

## 四、第三步：用 Trainer 把训练参数、日志和保存逻辑接起来

这一层在训练流程里负责的是：**把模型、数据和训练参数真正装进一个可跑的训练器。**

第六章用的是 `TrainingArguments + Trainer` 这条最典型的 `Transformers` 路线。骨架很简单：

```python
from transformers import TrainingArguments, Trainer, default_data_collator

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100,
    learning_rate=1e-4,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)
```

### 4.1 `TrainingArguments` 负责的是“训练策略的通用外壳”

比如：

- batch size
- 梯度累积
- 学习率
- 日志频率
- 保存策略
- gradient checkpointing

这些参数本身并不神秘，但放进 `TrainingArguments` 之后，训练脚本就会变得清晰很多：模型逻辑、数据逻辑和训练策略开始分层。

### 4.2 `Trainer` 真正统一掉的是训练脚手架

到了这里，很多通用工作都不需要自己再反复写：

- 前向和反向训练循环
- 日志打印
- checkpoint 保存
- tokenizer/data collator 接入
- 分布式训练接口对接

这也是第六章和第五章最明显的区别之一。第五章更像是在帮我理解训练闭环本身，而第六章开始让我看到：**当这些概念弄明白之后，真正实战里会交给框架去托住。**

这一部分真正该记住的点是：**Trainer 的价值不在于“替代理解”，而在于把已经理解的训练骨架抽成可复用模板。**

---

## 五、第四步：从 notebook 走向可长期运行的 DeepSpeed 训练脚本

如果说前面几节还在讲“如何用框架表达训练流程”，那么到了这里，重点就变成了：**怎么把这件事推进到更真实的多卡训练环境里。**

第六章进一步给出了：

- `pretrain.py`
- `finetune.py`
- `pretrain.sh`
- `finetune.sh`
- `ds_config_zero2.json`

这说明训练流程已经从“演示思路”切换到了“脚本化运行”。

`pretrain.sh` 的典型启动方式大致是这样：

```bash
deepspeed pretrain.py \
  --config_name autodl-tmp/qwen-1.5b \
  --tokenizer_name autodl-tmp/qwen-1.5b \
  --train_files autodl-tmp/dataset/pretrain_data/xxx.jsonl \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --output_dir autodl-tmp/output/pretrain \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --block_size 2048 \
  --bf16 \
  --gradient_checkpointing \
  --deepspeed ./ds_config_zero2.json
```

### 5.1 DeepSpeed 在这一章里主要承担两件事

第一件事，是让训练可以更自然地走向多卡。  
第二件事，是通过 ZeRO 等策略降低显存压力。

`ds_config_zero2.json` 里最值得记的关键词其实就是 `zero_optimization.stage = 2`。这说明第六章这里采用的是 `ZeRO-2`，也就是在参数之外，进一步去优化优化器状态和梯度的存储开销。

### 5.2 从 notebook 到脚本，真正增加的是“训练工程外壳”

比如在 `pretrain.py` 里，第六章补上了很多真实训练需要的东西：

- `HfArgumentParser` 解析命令行参数
- checkpoint 恢复
- 主进程数据预处理
- SwanLab 日志
- shell 层统一管理训练超参

这时再回头看，就会更容易区分两类内容：

- notebook 负责验证思路
- 脚本负责承接长时间训练

这一部分真正该记住的点是：**DeepSpeed 代表了第六章把训练从“教学演示”推向“可长期运行脚本”的那一步。**

---

## 六、第五步：进入 SFT 之后，关键差异集中在数据构造而不是 Trainer

这一部分我觉得是第六章最值得回头看的一段。因为它会把一个很重要的事实讲得非常清楚：

> **Pretrain 和 SFT 可以共享同样的 Trainer、DeepSpeed 和大部分训练框架，但它们的数据构造逻辑并不一样。**

原因很简单。  
Pretrain 训练的是整个文本分布，而 SFT 训练的是“给定指令后，assistant 应该怎样回答”。

所以进入 SFT 之后，最关键的问题就变成了：

- 对话怎么拼
- 角色怎么标
- 哪些 token 参与 loss
- padding 和 mask 怎么处理

第六章这里沿用了 Qwen 风格的 chat template，核心骨架大致像这样：

```python
im_start = tokenizer("<|im_start|>").input_ids
im_end = tokenizer("<|im_end|>").input_ids
IGNORE_TOKEN_ID = tokenizer.pad_token_id
nl_tokens = tokenizer("\n").input_ids

def preprocess(sources, tokenizer, max_len):
    input_ids, targets = [], []
    ...
    return dict(
        input_ids=torch.tensor(input_ids),
        labels=torch.tensor(targets),
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
```

### 6.1 这一节最核心的是 `labels` 的构造方式

对 SFT 来说，不是所有 token 都应该参与 loss。  
第六章里真正关键的设计，是：

- `system` 部分不参与拟合
- `user` 部分不参与拟合
- `assistant` 回复部分才是主要监督目标

这也是为什么 SFT 数据集看起来和 Pretrain 都是 `input_ids + labels + attention_mask`，但实际语义完全不同。

如果用一句更压缩的话来说：

> **Pretrain 的 `labels` 更像“整段文本的镜像”，SFT 的 `labels` 更像“只对回答区域打开监督信号”。**

### 6.2 `chat template` 是训练协议的一部分

到了 SFT，这件事会变得非常明显。  
像 `<|im_start|>`、`<|im_end|>`、角色标识、换行符这些 token，看起来只是格式细节，但它们其实在定义：

- 对话轮次边界
- 角色身份
- system/user/assistant 的结构关系

也正因为如此，第六章这里虽然还在讲“数据处理”，但本质上已经是在讲一个**对话训练协议**。

### 6.3 `attention_mask` 和 padding 依然不能忽视

到这里之后，padding 会影响：

- 哪些位置被模型看到
- 哪些位置不参与 loss
- batch 内不同长度样本如何对齐

这里也能顺手记一个细节：很多实现里 `IGNORE_TOKEN_ID` 常常会取 `-100`，因为这是 PyTorch 交叉熵里常见的忽略标签值。第六章的脚本写法更偏教学示意，但真正该记住的是“标签遮蔽”这个动作本身。

这一部分真正该记住的点是：**SFT 和 Pretrain 最大的不同，不在 Trainer，而在“谁参与监督”这条线。**

---

## 七、第六步：资源不够时，就把全量微调切到 LoRA

如果顺着前面的流程继续往下走，第六章最后补上的就是高效微调。

它先介绍了 Adapter、Prefix Tuning 等思路，最后把重心落到 LoRA 上。  
这一段之所以重要，是因为它把“为什么能省资源”和“具体怎么在工程里用”接到了一起。

LoRA 最核心的形式可以写成：

$$
W = W_0 + BA
$$

其中 $W_0$ 是冻结的原始权重，$A$ 和 $B$ 是低秩增量矩阵。训练时原始权重保持冻结，主要更新这两个低秩矩阵。

如果只抓工程落点，第六章这一节最值得记的就是 `peft` 的用法：

```python
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
```

### 7.1 LoRA 在这一章里负责的是“把可训练参数缩到更小”

它最适合的场景，是：

- 模型参数很大
- 训练资源有限
- 任务更偏行为适配或领域适配

这也是为什么第六章会把 LoRA 放在完整训练流程的后半段。因为在训练链路上看，它给 SFT 这类后训练阶段提供了一个更轻量的更新方式。

### 7.2 真正要记的是 peft 的抽象

第六章虽然解释了 LoRA 层替换的大致原理，但对工程实践来说，更关键的记忆点是：

- 目标层往往是 `q_proj`、`v_proj` 这类注意力投影层
- `LoraConfig` 用来描述 LoRA 超参
- `get_peft_model` 会把 LoRA 注入到原模型里

也就是说，在实战里更常见的工作，是搞清楚：

- 微调目标是什么
- LoRA 应该插到哪些层
- rank、dropout、alpha 怎么取

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/3-2.jpg" alt="LoRA 原理图" width="90%" />
  <p>图：LoRA 用低秩增量更新替代全量参数更新，是第六章高效微调部分的核心</p>
</div>

### 7.3 LoRA 的边界也要顺手记住

第六章最后其实已经点出了一个很重要的现实边界：  
LoRA 很适合任务适配，但如果目标是让模型真正吸收大量新知识，它往往不是最理想的路径。

所以从训练链路上看，更稳妥的理解应该是：

- Pretrain 更偏知识底座
- SFT 更偏助手行为塑形
- LoRA 更偏低成本适配

这一部分真正该记住的点是：**LoRA 是第六章里把后训练成本压下来的那条高效微调路线。**

---

## 八、写在最后：第六章真正搭起了怎样的训练骨架

如果把整章重新压缩一遍，我会把第六章总结成下面这条主线：

1. 用 `AutoConfig / AutoModel / AutoTokenizer` 接入模型和 tokenizer  
2. 用 `datasets.map` 和 `group_texts` 把预训练数据整理成 CLM block  
3. 用 `TrainingArguments + Trainer` 承接训练骨架  
4. 用 `DeepSpeed + ZeRO-2` 把训练推向更真实的多卡脚本环境  
5. 在 SFT 阶段把重心切到 `chat template + labels + attention_mask`  
6. 在资源受限时用 `LoRA + peft` 替代全量微调  

这样看下来，第六章真正搭起来的，是一套基于主流框架的训练流程骨架：

> **Pretrain -> SFT -> LoRA**

而且这条链路里，每一层的职责都更清楚了：

- 模型层负责承接结构和权重
- 数据层负责定义训练协议
- Trainer/DeepSpeed 负责承接训练基础设施
- LoRA 负责在后训练阶段压缩更新成本

至于 `6.4` 的偏好对齐部分，我更愿意把它看成一个全景补充。它帮我知道训练链路后面还会继续走向奖励模型、RLHF、DPO 这些更复杂的阶段，但第六章真正落到可复用实践上的重点，还是前面这条框架化训练主线。

这也是我现在回头看这一章时最清楚的收获：  
**它把大模型训练第一次放进了主流工程栈里。**

---

## 参考资料

- [Datawhale. Happy-LLM：第六章《大模型训练流程实践》](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter6/%E7%AC%AC%E5%85%AD%E7%AB%A0%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5.md)
- [Datawhale Happy-LLM 仓库](https://github.com/datawhalechina/happy-llm)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
