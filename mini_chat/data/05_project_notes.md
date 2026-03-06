# mini_chat 项目说明（用于面试）

这是一个基于 Streamlit 的小型 LLM 应用，包含：
- 对话 UI
- RAG：文档加载、切块、embedding、FAISS 检索、门控、拼接 prompt
- 记忆：长期记忆抽取与检索（可控、可持久化）
- 工程化：.env/.env.example、requirements.txt、缓存模型避免重复加载

## 当前已实现点（示例）
1. 使用 SentenceTransformer 生成 embedding，并用 FAISS IndexFlatIP 做余弦相似度检索（向量归一化）。
2. 增加防御式检索：top_k 不超过索引库大小；过滤 idx=-1；保证 docs 至少保留 top1 兜底。
3. 通过 st.cache_resource 缓存 embedding 模型，避免每次交互重复加载权重。

## 下一步方向
1. 引用展示：回答末尾展示引用 chunk 编号，提升可解释性。
2. 向量库持久化：保存 embeddings 与 FAISS index，二次启动秒开。
3. 记忆管理 UI：在侧边栏查看/删除/清空长期记忆。
