# 🧩 Engineering-Level Mini LLM App (RAG + Long-term Memory)

一个**可运行、可解释、带持久化**的小型大模型应用：

- **RAG 知识库问答**：从 `data/` 的 `.md/.txt` 文档中检索相关片段，把资料拼进 prompt，再让 LLM 生成回答
- **检索门控（Retrieval Gating）**：检索分数过低时拒答，降低“无资料胡编”的概率
- **长程记忆（Long-term Memory）**：从用户自述中抽取稳定信息并落盘保存（`storage/memory.json`）
- **对话历史压缩**：对话过长时自动摘要，保留关键事实，降低上下文长度
- **向量库持久化（FAISS on disk）**：向量索引与 embedding 缓存落盘，重启不必重算
- **流式输出**：Streamlit chat UI + streaming token

> 适合作为「大模型应用/平台/工程」方向实习项目：RAG / 向量检索 / 可靠性 / 记忆 / 工程化。

---

## ✨ 功能截图（建议你自己补）

- 运行后截图一张主界面
- 再录一段 20~40 秒的短视频/GIF（对投简历非常有效）

---

## ✅ 快速开始

### 1) 环境

```bash
python -m venv venv
# mac/linux
source venv/bin/activate
# windows
# venv\\Scripts\\activate

pip install -r requirements.txt
```

### 2) 配置 API Key

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

### 3) 运行

```bash
streamlit run app.py
```

---

## 📚 知识库数据

- 把你的资料放到 `data/` 目录：支持 `.md` / `.txt`
- 启动时会：读取 → 切块 → 去重 → 构建/加载 FAISS 索引

> 目前 demo 只做了 txt/md，后续可扩展 PDF / HTML / Docx 的解析。

---

## 📊 最小评测（强烈建议你做完并把结果写进简历）

项目自带一个非常轻量的 **retrieval + gating** 评测脚本：

```bash
make eval
# 或
python eval/run_eval.py --verbose
```

它会读取 `eval/qa.jsonl` 的小型 QA 集，输出：

- `hit@k`：正例问题检索命中率（基于 `must_contain` 关键词）
- `gate_accuracy`：该拒答的拒答、该回答的回答
- 正例/负例的 top1 相似度均值（帮助你调阈值）

评测结果会保存到：`eval/report.json`。

> ### Ablation: Retrieval Gating (abs_th)
>
> | abs_th | gate_accuracy |
> | -----: | ------------: |
> |   0.42 |          0.75 |
> |   0.60 |      **0.85** |
>
> `pos/neg top1 mean` 是检索分数分布统计，与门控无关，因此基本不随阈值变化。

---

## 🧠 长期记忆

- 当用户输入包含「我叫/我是/目标/偏好/记住/我喜欢…」等表达时，会触发记忆抽取
- 记忆以列表形式持久化到 `storage/memory.json`
- Sidebar 支持展示与清空

---

## 🧱 工程结构（推荐面试时这样讲）

```
app.py                 # Streamlit 入口（对话流 + 逻辑编排）
core/
  config.py            # 环境变量配置
  llm_client.py         # OpenAI SDK 封装（兼容 DeepSeek 等 OpenAI-like API）
  router.py             # 意图路由（记忆查询/记忆写入/默认RAG）
  history.py            # 历史对话压缩
  paths.py              # 存储路径
memory/
  extractor.py          # LLM 结构化抽取长期记忆
  store.py              # 记忆落盘/去重/检索
rag/
  loader.py             # 文档加载
  chunker.py            # 切块（overlap）
  vector_store.py       # SentenceTransformer + FAISS + 落盘持久化
  query.py              # query 归一化
  gate.py               # 检索门控
  rerank.py             # 词面关键词兜底 rerank
  prompt.py             # RAG prompt 拼接（强约束 + 引用编号）
  verify.py             # (debug) LLM 自检：回答是否仅来自资料
ui/
  sidebar.py            # sidebar 组件
```

---

## 🧪 Roadmap（强烈建议你按这个路线做完 = 实习简历很能打）

1. **README 完整化 + Demo 视频 + 截图**（最重要）
2. **可解释引用**：把检索到的 chunk 用 expander 展示，引用编号对齐
3. **KB 可热更新**：支持用户上传文件并增量构建索引
4. **Memory 检索升级**：从“字符串匹配”升级到“embedding 检索”（更像工业实现）
5. **可靠性**：verify 不合格时自动重写/拒答，形成闭环
6. **评测脚本**：准备 30~50 条 QA 测试集，输出命中率/拒答率/合格率（可写进简历）
7. **Docker + CI**：Dockerfile、单元测试、lint（展示工程素养）

---

## 🧾 你简历上可以怎么写（示例）

- 从 0 搭建 **RAG 问答系统**：实现文档切块、向量检索、FAISS 持久化索引、检索门控与引用输出
- 设计 **长期记忆模块**：基于 LLM 抽取稳定用户信息并落盘，支持检索与多轮对话背景注入
- 增加 **对话压缩** 与 **自检验证**：降低上下文长度并减少无依据回答（可在评测后给出量化指标）

---

## License

MIT (可选)
