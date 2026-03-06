# Streamlit 工程注意事项

Streamlit 的脚本会在交互时 rerun（从上到下重新执行）。
如果你每次 rerun 都重新加载大模型或重新构建索引，会非常慢。

## 常见做法
- 使用 st.cache_resource 缓存重资源（embedding 模型、向量索引）
- 文档数据未变化时，避免重新 encode

## 如何判断缓存是否生效
如果每次提问都打印 Loading weights，就说明模型加载没有缓存。
缓存正确时：
- 第一次启动会加载一次
- 后续提问不会重复加载权重

## 环境一致性
在 Windows 上常见“终端解释器”和“PyCharm 运行解释器”不一致。
建议统一使用 venv，并在终端用 python -m pip 与 python -m streamlit 运行。
