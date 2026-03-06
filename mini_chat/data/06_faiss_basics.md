# FAISS 检索基础（IndexFlatIP）

FAISS 是常用的向量检索库。最简单的索引是 IndexFlat 系列：
- IndexFlatL2：使用 L2 距离
- IndexFlatIP：使用内积（Inner Product）

如果你的向量做了 normalize（L2 归一化），那么：
- 内积（IP）等价于余弦相似度（cosine similarity）
因此很多语义检索会使用：normalize + IndexFlatIP。

## 常见坑
1) top_k 大于索引库大小（ntotal）：
当 k > ntotal 时，一些索引会用占位结果填充，导致出现无效 idx 或极小值分数（以前你遇到的 -3.4e38 就属于这种症状之一）。
解决：k = min(top_k, ntotal)，并过滤 idx == -1。

2) embeddings dtype：
FAISS 通常要求 float32，如果传 float64 可能慢或出错。建议 encode 后统一 astype("float32")。

3) 归一化：
如果你没归一化却用 IP，会偏好向量长度而不是语义相近。
