import streamlit as st

def show_docs_debug(docs: list[str], max_chars: int = 140):
    st.write("### DEBUG docs")
    for i, d in enumerate(docs, 1):
        st.write(i, d[:max_chars])
