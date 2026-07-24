---
title: "从 Prompt 到可审计系统：结构化 LLM 决策的工程笔记"
description: "整理候选约束、结构化输出、Critic 自纠、冻结回放、推测解码、Prefix Cache 和并发调优中的通用经验"
date: 2026-07-24
slug: structured-llm-decision-system-review
image:
categories:
    - LLM-Learning
tags:
    - LLM
    - LLM Engineering
    - Evaluation
    - Retrieval
    - Structured Output
    - Speculative Decoding
    - vLLM
    - MTP
math: false
---

最近一个月，我一直在整理一类结构化 LLM 决策系统。它接收自然语言和结构化字段，在有限标签中做选择；证据不足时，系统要能转人工或者明确放弃自动判断。

这篇只保留能公开讨论的方法，不涉及具体行业、业务字段、数据规模和实测数字。真正让我花时间的也不是某一句 Prompt，而是几个更基础的问题：候选从哪里来，怎样限制模型输出，第二次调用为什么可能和第一次不同，服务变快以后质量会不会发生变化。

## 一、先把“生成答案”改成“受限决策”

开放式生成很灵活，直接放进业务流程却容易出问题：模型可能编出不存在的标签，也可能忽略方向、阶段之类的约束。即使返回了合法 JSON，字段值仍然可能选错。

我更习惯把流程拆成下面这样：

```text
原始输入
  -> 清洗和确定性规则
  -> 检索候选
  -> LLM 提取受控事实
  -> 在候选 ID 中选择
  -> 程序检查 + 必要时语义复核
  -> 自动通过 / 人工复核 / 放弃自动判断
```

这类结构可以用在文档分类、工单路由、内容审核或其他高风险分类任务中。模型只负责一个边界清楚的局部判断：

```text
decision = model(facts, candidates, policy)
```

`facts` 是当前样本中可验证的事实，`candidates` 是允许选择的集合，`policy` 描述决策边界。三者最好分开组织。全塞进一段自然语言里，后面很难知道模型到底忽略了哪一部分。

### JSON 合法，只解决了第一层问题

结构化输出有两种常见做法。一种是在 Prompt 里要求返回 JSON，解析失败后重试；另一种是在解码阶段按照 JSON Schema 屏蔽非法 token，也就是 constrained decoding。

[OpenAI 对 Structured Outputs 的说明](https://openai.com/index/introducing-structured-outputs-in-the-api/)提到，约束解码可以保证输出符合 schema，却不能保证对象里的值正确。一个答案可以格式完美，同时语义完全错误。

所以校验至少要分成两层：

```text
结构校验：字段是否齐全、类型是否正确、枚举是否合法
领域校验：候选是否存在、事实是否冲突、决策是否越过业务边界
```

模型最好返回候选里的稳定 ID，不要重新生成标签名称。解释可以是一小段文本，真正参与程序路由的状态尽量使用枚举、布尔值和受控数组。

还有一个常被漏掉的分支：模型拒绝回答，或者因为长度限制没有生成完整对象。结构化输出接口通常会单独暴露 refusal 和 finish reason，程序必须处理，不能把缺失字段补成一个看似正常的结果。

## 二、检索的价值，是把问题变小

在分类系统里，检索的任务不只是补充知识。它还负责缩小模型的选择空间。

| 环节 | 目标 | 常用观察指标 |
|---|---|---|
| 确定性规则 | 接住边界清楚的输入 | 覆盖率、冲突率 |
| 候选召回 | 让正确标签进入 top-k | Recall@K |
| 候选排序 | 把更相关的候选排在前面 | MRR、NDCG |
| LLM 决策 | 结合事实做最终选择 | 自动决策精度、放弃率 |

召回决定了系统的理论上限。正确答案没有进入候选池，后面的 LLM 再强也选不到。反过来，Recall@K 很高也不代表最终结果一定更好。候选太多会增加 token，相近标签之间也会互相干扰。

候选数量还会影响上下文利用率。[Lost in the Middle](https://arxiv.org/abs/2307.03172) 发现，相关信息位于长上下文中部时，模型表现可能明显下降。“窗口装得下”和“模型用得好”并不是一回事。

因此 top-k 应该被当作需要评测的超参数：

```text
k 太小：正确答案可能没有被召回
k 太大：上下文变长，候选彼此干扰
合适的 k：兼顾 Recall@K 和最终决策质量
```

BM25、Embedding 和 Reranker 适合解决不同的问题。BM25 对关键词、编号和专有名词敏感；Embedding 能处理同义表达；Reranker 在较小的候选集上做更细的 query-document 交互。常见做法是先用便宜的检索器召回，再把少量候选交给 Reranker 或 LLM。

候选来源也值得保留。某个候选来自规则、稀疏检索还是向量检索，应该跟着结果进入 trace。误判发生时，这能帮助判断该调整召回还是决策节点。

## 三、中间状态也是接口

多节点流程里，一个事实分析节点可能先把原文压缩成摘要，后续节点再根据摘要做判断。这样很方便，也很容易积累漂移。

自然语言摘要是一种有损压缩。它可能省略否定和范围，也可能把推断写成事实。后续模型并不知道这句话来自原始输入还是上一个模型的猜测，只会把它当作新的上下文继续推理。

更稳的中间协议可以写成：

```json
{
  "direction": "outbound",
  "entity_type": "organization",
  "stage": "execution",
  "has_supporting_document": true,
  "evidence": ["原文片段 A", "输入字段 B"],
  "uncertain_fields": []
}
```

这里最重要的不是字段叫什么，而是把“输入事实”“模型推断”和“不确定项”分开。自由文本仍然可以留在 `reason` 中供人阅读，但不要让它悄悄控制程序分支。

从工程角度看，LLM 节点和普通服务没有区别：输入、输出、版本和失败行为都需要契约。模型生成的文字很顺，所以接口漂移有时比普通类型错误更难发现。

## 四、Critic 能用，但要先告诉它查什么

多轮反思看起来很自然：先生成答案，再让模型审查，发现问题后重新生成。研究对这种方法给出了不同结果。

[Self-Refine](https://arxiv.org/abs/2303.17651) 在多种任务上观察到自反馈和迭代改写带来的提升；[Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) 则发现，没有外部反馈时，模型可能找不到自己的推理错误，修改后表现甚至会下降。

这两个结论并不冲突。润色文本、补充遗漏，和定位一个未知的逻辑错误不是同一道题。Critic 如果只收到一句“请严格审查”，很可能重复第一次判断，也可能为了表现得更严格而修改原本正确的答案。

我更认可有边界的审查：

1. 程序先检查 ID、枚举、互斥条件等硬约束。
2. 规则无法判断时，再调用一次语义 Critic。
3. Critic 只检查预先定义的错误类型。
4. 高风险状态变化需要独立证据或短确认。
5. 达到调用上限后转人工，不允许无限反思。

Critic 真正需要的是外部锚点。程序规则、候选约束、错误位置和独立检索证据都算锚点。什么都不给，只让模型“再想想”，效果很难预测。

还可以把“发现错误”和“修正错误”拆成两个节点。已有研究表明，模型可能具备修正已定位错误的能力，却不擅长自己找到错误位置。拆开后，每个节点的评测也更清楚。

## 五、评测不能只留一个 accuracy

高风险分类中，不同错误的代价差很多。自动给出错误结果和把正确样本转给人工，不能算成同一种失败。

一组更实用的指标包括：

- 总体准确率；
- 自动决策精度和错误自动放行数；
- 人工复核率与错误放弃数；
- 最终状态分布；
- 各节点调用数、P50/P95 延迟、输入与输出 token；
- schema retry、服务 retry、超时和错误类型。

如果业务更重视避免错误自动决策，就应该给自动决策精度设置独立门槛，而不是让总体 accuracy 把它平均掉。可以把目标写成简单的成本函数：

```text
total_cost
  = false_accept * 高风险成本
  + false_abstain * 机会成本
  + manual_review * 人工成本
  + latency * 服务成本
```

这些权重来自业务风险，不应由模型指标替代。

### 全链路评测和冻结回放回答不同问题

全链路评测回答“这个版本能不能用”。冻结回放回答“变化从哪里产生”。

冻结回放会保存某个节点的完整输入，固定 Prompt、候选和依赖，只替换模型或推理参数。如果冻结输入仍然漂移，问题更可能在模型或运行时；如果冻结后稳定、全链路却不稳定，就应检查上游状态。

同一实验也需要重复运行。除了均值，还应观察最差轮次、方差和变化样本交集。一直失败的样本容易定位；偶尔正确、偶尔自信做错的样本更值得警惕。

## 六、推理优化要分清 prefill 和 decode

一个 LLM 请求大致有两个阶段：

```text
prefill：处理整段输入，建立 KV cache 或模型状态
decode：逐 token 生成输出，过程更串行
```

上下文、候选和示例过长，主要增加 prefill；输出理由太长、反复 repair，主要增加 decode 和调用次数。优化前先拆开这些指标，不然很容易在错误的阶段花时间。

### Prefix cache 只节省共享前缀的 prefill

[vLLM 的 Automatic Prefix Caching 文档](https://docs.vllm.ai/en/v0.15.0/features/automatic_prefix_caching/)明确说明，APC 会复用相同前缀的 KV cache，从而跳过共享部分的 prefill，但不会减少新 token 的 decode 时间。

它更适合长公共前缀、多轮对话，或者对同一文档的重复查询。如果瓶颈是长输出或重复调用，prefix cache 的帮助有限。

Prompt 的排列也会影响缓存命中。稳定的系统指令、schema 和公共规则可以放在前面，动态事实与候选放在后面。不过这个建议仍需在具体运行时上验证。Prefix cache、推测解码、模型架构和并发调度可能互相影响，任何组合都不能只凭单项基准上线。

### MTP 和推测解码

普通自回归解码一次只确定一个 token。推测解码先草拟多个 token，再由目标模型一次验证。经典的 [Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) 使用接受/拒绝采样保持目标模型的输出分布，因此草拟模型不会直接替代目标模型。

MTP 使用模型自带的 multi-token prediction head 预测后续 token。vLLM 的 [MTP 文档](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/algorithms/mtp/)也把它归在 speculative decoding 中。

评估 MTP 时至少要记录：

```text
draft acceptance rate
mean accepted length
inter-token latency
request throughput
P50 / P95 latency
端到端质量指标
```

接受率高不等于服务一定更快。草拟和验证本身也有计算成本；高并发下，普通 decode 已经能组成较大的 batch，MTP 的相对收益可能缩小。

### 并发不是越大越好

vLLM 使用 PagedAttention 和连续批处理提高显存利用率与吞吐。[PagedAttention 论文](https://arxiv.org/abs/2309.06180) 讨论了如何减少动态 KV cache 的碎片和重复保存，从而容纳更有效的 batch。

但 batch 增大后，每一步 decode 会有更多序列竞争计算和内存带宽。`max-num-seqs`、客户端并发和吞吐之间通常不是单调关系，需要逐档压测。

排查服务瓶颈时，可以先做这样的判断：

```text
队列高：检查调度容量、限流或服务副本
prefill 高：检查上下文、候选数量和前缀复用
decode 高：检查输出长度、MTP 和重复调用
GPU 不忙但流程慢：检查网络、串行节点和任务编排
```

GPU 利用率本身说明不了请求是否高效。它需要和队列、延迟、吞吐、KV cache 使用量一起看。

## 七、一次结果需要保存哪些版本

模型版本只是复现条件的一部分。结果还会受到检索索引、标签集合、参考数据、策略配置、Prompt、采样参数和运行时版本影响。

对外部依赖，可以先规范化 JSON，再计算内容哈希；一次任务只引用不可变的快照 ID 和哈希，不在执行中途重新获取可能变化的数据。

我现在会把一次 LLM 实验看成下面这个元组：

```text
run = (
  code_commit,
  model_and_runtime,
  prompt_version,
  decoding_config,
  dependency_snapshot,
  evaluation_dataset,
  random_and_request_metadata
)
```

少了其中一项，复现失败时都可能说不清原因。即使温度设为 0，也不要假设系统绝对确定。并行归约、批处理顺序、服务版本和上游自然语言状态仍可能产生差异。

运行配置可以进入版本控制，密钥不可以。可复现实验和 secret 管理是两件事。

## 八、通用 Prompt 不应该保存本地政策

领域示例经常不只是在教模型怎样推理，还可能暗中携带业务政策。直接删掉示例，模型失去的也许不是语言能力，而是决策边界。

更清楚的拆法是：

```text
通用推理层：怎样阅读事实、比较候选、处理冲突
领域知识层：常见概念、关系和表达方式
本地政策层：允许的边界、例外规则和风险阈值
```

推理方法可以复用，政策应显式配置和版本化。所谓泛化，也需要在新的标签集合、独立标注和不同输入分布上验证。把特定词汇从 Prompt 里删掉，只能说明文本变得更抽象，不能说明系统更通用。

## 以后再做类似系统，我会先检查这些

- 正确答案不在候选里时，系统是否会明确停止？
- 模型返回的是稳定 ID，还是重新生成标签名称？
- schema 合法后，是否还有领域校验？
- 中间节点传递的是受控状态，还是会漂移的摘要？
- Critic 是否知道具体要检查哪类错误？
- 自动做错和转人工是否使用不同门槛？
- 能否冻结任意节点输入，单独回放模型与运行时？
- 性能数据是否拆开 queue、prefill、decode 和调用次数？
- 每次实验能否还原代码、模型、Prompt、依赖和评测集？
- 通用方案是否在新的数据分布上验证过？

一个月前，我更关注模型能不能给出正确答案。现在我更在意，当模型给出答案时，系统有没有足够证据决定自动接受、转给人工，或者承认信息不足。这个问题没什么神秘的，却决定了 LLM 能不能进入真实流程。

## 延伸阅读

- [Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)
- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.15.0/features/automatic_prefix_caching/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
