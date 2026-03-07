import streamlit as st

def render_memory_sidebar(memory_store):
    st.sidebar.header("🧠 Long-term Memory")
    mems = memory_store.list_memories()
    if mems:
        for i, m in enumerate(mems):
            st.sidebar.write(f"{i}. {m}")
    else:
        st.sidebar.caption("暂无长期记忆")

    if st.sidebar.button("清空长期记忆"):
        memory_store.clear()
        st.sidebar.success("已清空")
