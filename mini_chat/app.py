import os
import re
import streamlit as st

from core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MODEL_NAME,
    TOP_K,
    TOP_SIM_THRESHOLD,
    MAX_RAW_MESSAGES,
    MAX_TURNS,
    require_key,
)
from core.history import compress_old_messages, get_recent_messages
from core.llm_client import LLMClient
from core.paths import MEMORY_PATH
from core.router import Intent, route_intent, is_memory_write,is_memory_query

from memory.extractor import MemoryExtractor
from memory.store import MemoryStore

from rag.chunker import split_text
from rag.gate import should_reject
from rag.loader import load_documents_from_folder
from rag.prompt import PromptBuilder
from rag.query import normalize_query
from rag.rerank import select_docs_with_fallback
from rag.vector_store import build_vector_store
from rag.verify import verify_answer

from ui.sidebar import render_memory_sidebar


@st.cache_data(show_spinner=False)
def load_and_chunk_kb(data_dir: str = "data", chunk_size: int = 220, overlap: int = 50) -> list[str]:
    """Load markdown/txt files and split into deduplicated chunks."""
    if not os.path.isdir(data_dir):
        return []

    # list of (source_name, text)
    raw_docs = load_documents_from_folder(data_dir)

    documents: list[str] = []
    for source_name, doc in raw_docs:
        for i, c in enumerate(split_text(doc, chunk_size=chunk_size, overlap=overlap), 1):
            c = (c or "").strip()
            if len(c) >= 10:
                # Add light metadata for explainability / eval.
                documents.append(f"[source={source_name}][chunk={i}]\n{c}")

    # dedup (keep order)
    seen = set()
    dedup_docs: list[str] = []
    for d in documents:
        if d not in seen:
            seen.add(d)
            dedup_docs.append(d)

    return dedup_docs
@st.cache_resource
def get_vs(documents):
    return build_vector_store(tuple(documents))

def main():
    st.set_page_config(page_title="Mini LLM App", page_icon="🧩", layout="centered")
    st.title("🧩 Engineering-Level Mini LLM App")

    # fail fast
    try:
        require_key()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # init session state
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history_summary", "")

    # init core objects
    llm = LLMClient(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME)
    memory_store = MemoryStore(MEMORY_PATH)

    # sidebar
    st.sidebar.header("⚙️ Settings")
    show_debug = st.sidebar.checkbox("显示 Debug", value=False)

    st.sidebar.divider()
    render_memory_sidebar(memory_store)

    # load KB + vector store
    documents = load_and_chunk_kb("data", chunk_size=220, overlap=50)
    if not documents:
        st.warning("知识库为空：请在 data/ 放入 .md 或 .txt 文件。")
        st.stop()

    vs = get_vs(tuple(documents))

    # show summary + history
    st.subheader("📜 Conversation History Summary")
    if st.session_state.history_summary:
        st.write(st.session_state.history_summary)
    else:
        st.caption("暂无摘要（对话达到阈值后会自动压缩）")

    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    prompt = st.chat_input("Ask something") or ""
    if not prompt.strip():
        return

    # user msg
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ---- (1) 路由意图（先路由，避免后续写入/检索逻辑重复触发） ----
    intent = route_intent(prompt)
    print("[DEBUG] intent =", intent)
    # ✅ 路由兜底：关键词命中就强制当作记忆查询
    if is_memory_query(prompt):
        intent = Intent.MEMORY_QUERY

    # ---- (1.1) 记忆查询：优先处理，直接 return（避免误触发写入/走RAG） ----
    if intent == Intent.MEMORY_QUERY:
        mems = memory_store.list_memories()
        if not mems:
            reply = "我现在还没有记录到你的长期记忆。你可以直接说：'记住：…' 或 '我喜欢…'。"
        else:
            bullet = "\n".join([f"- {m}" for m in mems])
            reply = f"我目前记住的长期信息有：\n{bullet}"
        st.chat_message("assistant").write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    # ---- (1.2) 记忆写入：只抽取一次、只写一次 ----
    # 显式写入意图：用户说“记住：…” “请记住…”
    explicit_write = (intent == Intent.MEMORY_WRITE) or is_memory_write(prompt)

    # 隐式写入触发：用户自我信息（可按需增删关键词）
    implicit_write = any(k in prompt for k in [
        "我叫", "我是", "目标", "希望", "偏好", "从现在开始", "记住",
        "我喜欢", "我不喜欢", "我想", "我目前", "我在"
    ])

    memories = []
    if explicit_write or implicit_write:
        # 先用 LLM 抽取
        memories = MemoryExtractor.extract(llm.client, prompt, MODEL_NAME) or []

        # 兜底：LLM 抽取为空时，规则抽取“我叫X”
        if not memories:
            m = re.search(r"我叫([\u4e00-\u9fa5A-Za-z0-9]{1,20})", prompt)
            if m:
                memories = [f"我叫{m.group(1)}"]

        # 去重写入（MemoryStore.add_memory 内部也会去重）
        wrote = []
        for m in memories:
            if m and m.strip():
                memory_store.add_memory(m.strip())
                wrote.append(m.strip())
        if wrote:
            print("[DEBUG] wrote memories:", wrote)
            print("[DEBUG] memory file:", os.path.abspath(MEMORY_PATH))

    # 如果是“显式写入”，给用户一个确认回复并 return（避免继续走RAG）
    if explicit_write:
        if memories:
            bullet = "\n".join([f"- {m}" for m in memories])
            reply = f"好的，我记住了：\n{bullet}"
        else:
            reply = "收到，我记住了。"
        st.chat_message("assistant").write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return
    # ---- (1) 记忆逻辑结束 ----

    # 3) compress long history
    if len(st.session_state.messages) > MAX_RAW_MESSAGES:
        old_part = st.session_state.messages[:-MAX_TURNS]
        new_part = st.session_state.messages[-MAX_TURNS:]
        summary = compress_old_messages(llm.client, MODEL_NAME, old_part)
        st.session_state.history_summary += ("\n" + summary) if st.session_state.history_summary else summary
        st.session_state.messages = new_part

    # assistant
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        related_memories = memory_store.search(prompt)
        memory_context = "\n".join(related_memories) if related_memories else ""
        # ---------- ✅ 记忆优先兜底：防止被 RAG 门控挡住 ----------
        # 这类问题优先用长期记忆回答（名字/身份/偏好/目标）
        memory_q = any(k in prompt for k in [
            "我的名字", "我叫什么", "我叫啥", "名字是什么", "我是谁",
            "我的目标", "我的偏好", "我喜欢什么", "我不喜欢什么"
        ])

        if memory_q:
            # 命中记忆就直接答，不走RAG
            hits = memory_store.search(prompt)
            if hits:
                # 针对“名字”再精一点：优先挑包含“名字/我叫”的记忆
                if any(k in prompt for k in ["名字", "叫什么", "我叫", "称呼"]):
                    chosen = None
                    for m in hits:
                        if ("名字" in m) or ("我叫" in m):
                            chosen = m
                            break
                    chosen = chosen or hits[0]
                    reply = f"我记得你说过：{chosen}"
                else:
                    bullet = "\n".join([f"- {m}" for m in hits])
                    reply = f"我记得这些相关信息：\n{bullet}"

                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                return
        # ---------- 记忆优先兜底结束 ----------
        # RAG retrieve
        query_for_retrieve = normalize_query(prompt)
        results = vs.retrieve(query_for_retrieve, top_k=TOP_K)
        results = sorted(results, key=lambda x: x[0], reverse=True)

        if show_debug:
            st.write(f"**[DEBUG] query_for_retrieve = {query_for_retrieve}**")
            st.write("**[DEBUG] top results:**", [(round(s, 3), d[:60]) for s, d in results[:5]])

        if should_reject(results, abs_th=0.42, gap_th=0.0):
            msg = "我不知道，资料中没有相关信息。"
            st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        docs = select_docs_with_fallback(
            results=results,
            query=query_for_retrieve,
            top_sim_threshold=TOP_SIM_THRESHOLD,
            min_keep=1,
        )
        if not docs:
            msg = "我不知道，资料中没有相关信息。"
            st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        # prompt build
        system_prompt = PromptBuilder.build_rag_prompt(docs)
        messages_for_llm = [{"role": "system", "content": system_prompt}]

        mem_prompt = PromptBuilder.build_memory_prompt(memory_context)
        if mem_prompt:
            messages_for_llm.append(mem_prompt)

        recent_msgs = get_recent_messages(st.session_state.messages, MAX_TURNS)
        messages_for_llm.extend(recent_msgs)

        if st.session_state.history_summary:
            messages_for_llm.append(
                {
                    "role": "system",
                    "content": "以下是历史对话摘要，仅供背景参考：\n" + st.session_state.history_summary,
                }
            )

        # stream generation
        for chunk in llm.chat_stream(messages_for_llm):
            if chunk.choices[0].delta.content:
                full_reply += chunk.choices[0].delta.content
                placeholder.markdown(full_reply)

        # verify (debug-only, do not block)
        if show_debug:
            ok = verify_answer(llm.client, MODEL_NAME, full_reply, docs)
            st.write(f"**[DEBUG] verify: {ok}**")

            with st.expander("查看本次引用资料"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**[{i}]** {d}")

    st.session_state.messages.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    main()
