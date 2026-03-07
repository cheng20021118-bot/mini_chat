# 🧩 Engineering-Level Mini LLM App
**RAG + Long-term Memory + Retrieval Gating + FAISS Persisted (Streamlit)**

一个面向「大模型应用/工程」实习的 Mini LLM App：可运行、可解释、可复现，并包含最小评测与门控对照实验。

## Highlights
- ✅ **RAG 知识库问答**：从 `data/` 的 `.md/.txt` 文档检索片段 → 拼 Prompt → 生成回答（支持引用）
- ✅ **检索门控（Retrieval Gating）**：检索分数过低时拒答，降低无资料胡编
- ✅ **长程记忆（Long-term Memory）**：抽取用户稳定信息并落盘（`storage/memory.json`），支持记忆优先问答
- ✅ **对话历史压缩**：超长对话自动摘要，减少上下文长度
- ✅ **向量库持久化（FAISS on disk）**：索引与 embedding 落盘，重启无需重建
- ✅ **Streamlit Chat UI + 流式输出**

---

## Demo（建议把 GIF/视频放这里）
> 你可以用 `assets/demo.gif` 或网盘/B站链接替换下面占位。

- 🎬 Demo Video/GIF: **TODO**
- 🖼️ Screenshot: **TODO**

**推荐演示脚本（20~40 秒）**
1. 问一个 `data/` 中能回答的问题 → 展示回答 + 引用
2. 问一个明显无关的问题 → 触发拒答
3. 输入：`记住：我叫张三` → 再问：`我的名字是什么？`

---

## Quickstart

### 1) 环境安装
```bash
python -m venv venv
# mac/linux
source venv/bin/activate
# windows
# venv\Scripts\activate

pip install -r requirements.txt
```

### 2) 配置 API Key
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

## Knowledge Base（知识库数据）
- 把资料放到 `data/`：支持 `.md` / `.txt`
- 启动时会：读取 → 切块 → 去重 → 构建/加载 FAISS 索引

---

## Evaluation（最小评测 + 门控对照）
项目自带轻量的 **retrieval + gating** 评测脚本（读取 `eval/qa.jsonl`）：

```bash
make eval
# 或
python -m eval.run_eval
```

输出指标：
- `hit@k`：正例问题检索命中率（基于 `must_contain` 关键词代理）
- `gate_accuracy`：该拒答的拒答、该回答的回答
- `pos/neg top1 mean`：检索 top1 相似度均值（用于调阈值）

评测结果保存：`eval/report.json`

### Ablation: Retrieval Gating (abs_th)
评测集规模：total=20, pos=14, neg=6

| abs_th | gate_accuracy | hit@k |
| -----: | ------------: | ----: |
|   0.42 |          0.75 |  1.00 |
|   0.60 |      **0.85** |  1.00 |

Artifacts:
- `eval/report.json`
- `eval/out_th_0.42.txt`
- `eval/out_th_0.60.txt`

> 注：`pos/neg top1 mean` 是检索分数分布统计，与门控阈值无关，因此基本不随阈值变化。

---

## Long-term Memory（长期记忆）
- 当用户输入包含「我叫/我是/目标/偏好/记住/我喜欢…」等表达时触发记忆抽取
- 记忆以列表形式持久化到 `storage/memory.json`
- Sidebar 支持展示与清空
- 对“名字/身份/偏好”等问题，优先使用记忆回答（避免被 RAG 门控误拒）

---

## Architecture（面试建议这样讲）
```
User Query
→ (optional) Memory Extract/Store
→ Memory-first QA (name/identity/preference)
→ RAG Retrieve (FAISS)
→ Retrieval Gating (abs_th)
→ Rerank + Prompt Build
→ LLM Generation (stream)
→ (debug) Verify Answer
```

---

## Project Structure
```
app.py                 # Streamlit 入口（对话流 + 编排）
core/
  config.py            # 环境变量配置
  llm_client.py        # OpenAI SDK 封装（兼容 DeepSeek 等 OpenAI-like API）
  router.py            # 意图路由（记忆查询/记忆写入/默认RAG）
  history.py           # 历史对话压缩
  paths.py             # 存储路径
memory/
  extractor.py         # LLM 结构化抽取长期记忆
  store.py             # 记忆落盘/去重/检索
rag/
  loader.py            # 文档加载
  chunker.py           # 切块（overlap）
  vector_store.py      # SentenceTransformer + FAISS + 落盘持久化
  query.py             # query 归一化
  gate.py              # 检索门控
  rerank.py            # 词面关键词兜底 rerank
  prompt.py            # RAG prompt 拼接（强约束 + 引用编号）
  verify.py            # (debug) LLM 自检：回答是否仅来自资料
eval/
  qa.jsonl             # 最小评测集
  run_eval.py          # 评测脚本
```

---

## Roadmap
- [ ] Demo GIF/视频 + 截图补齐（提升投递命中率）
- [ ] 引用可解释：expander 展示命中 chunk 与引用编号对齐
- [ ] KB 热更新：上传文件增量入库
- [ ] Memory 检索升级：字符串匹配 → embedding 检索
- [ ] 可靠性闭环：verify 不合格时自动重写/拒答
- [ ] 扩大评测集（30~50 QA）+ 输出更多可靠性指标
- [ ] Docker + CI（tests + lint）

---

## Resume Bullets（可直接用）
- 搭建 RAG 问答系统：实现文档切块、向量检索、FAISS 持久化索引、检索门控与引用输出
- 设计长期记忆模块：基于 LLM 抽取稳定用户信息并落盘，支持记忆优先问答与对话注入
- 构建最小评测脚本并进行门控对照实验：`gate_accuracy 0.75 → 0.85`（abs_th: 0.42 → 0.60）

---

## License
MIT
