# RAG（Retrieval-Augmented Generation）流程

RAG 的核心思想是：先检索外部资料，再把资料作为上下文提供给大模型生成答案。

## 标准流程
1. 文档加载：读取 data 目录的文本、md、pdf（demo 阶段先只做 txt/md）。
2. 切块：把长文切成 chunk（例如 300~500 字符，overlap 50~100）。
3. 向量化：对每个 chunk 生成 embedding。
4. 建索引：把向量加入向量库（如 FAISS）。
5. 查询检索：对用户 query 生成 embedding，在向量库里找 top_k chunk。
6. Prompt 拼接：把检索到的 chunk 拼成“资料区”，并加入回答规则。
7. 生成：调用 LLM 输出回答。
8. 可选校验：让模型做 self-verification，检查回答是否仅来自资料。

## Retrieval Gating（检索门控）
如果检索结果最高分也很低，说明“资料不相关”，就应该拒答或提示用户补充资料。
但要注意：在数据量很少时，分数本身会抖动，需要兜底策略（至少保留 top1）。

## 失败模式
- 检索没命中：资料不足或切块策略不合理。
- 资料命中但生成胡编：prompt 约束不够强，或资料表达不清晰。
- 引用不准：没有对 chunk 做编号与引用展示，用户无法验证。
