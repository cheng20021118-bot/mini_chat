import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag.chunker import split_text
from rag.gate import should_reject
from rag.loader import load_documents_from_folder
from rag.query import normalize_query
from rag.vector_store import VectorStore


@dataclass
class QACase:
    id: str
    query: str
    expect_reject: bool
    must_contain: Optional[str] = None
    note: str = ""


def load_qa_jsonl(path: str) -> List[QACase]:
    cases: List[QACase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)
            cases.append(
                QACase(
                    id=str(obj.get("id")),
                    query=str(obj.get("query")),
                    expect_reject=bool(obj.get("expect_reject")),
                    must_contain=obj.get("must_contain"),
                    note=str(obj.get("note") or ""),
                )
            )
    return cases


def build_kb_chunks(
    data_dir: str,
    chunk_size: int = 220,
    overlap: int = 50,
    min_len: int = 10,
) -> List[str]:
    """Build KB chunks with light metadata prefix.

    The metadata makes debugging / eval / citations easier.
    """
    docs = load_documents_from_folder(data_dir)
    chunks: List[str] = []
    for source_name, text in docs:
        for i, c in enumerate(split_text(text, chunk_size=chunk_size, overlap=overlap), 1):
            c = (c or "").strip()
            if len(c) < min_len:
                continue
            chunks.append(f"[source={source_name}][chunk={i}]\n{c}")

    # dedup keep order
    seen = set()
    out: List[str] = []
    for c in chunks:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def evaluate(
    vs: VectorStore,
    cases: List[QACase],
    top_k: int = 8,
    abs_th: float = 0.42,
    verbose: bool = False,
) -> Dict[str, Any]:
    total = len(cases)
    pos_total = 0
    neg_total = 0
    gate_correct = 0
    hit_at_k = 0

    # Some extra stats to help tuning threshold
    pos_top1_scores: List[float] = []
    neg_top1_scores: List[float] = []

    for c in cases:
        q = normalize_query(c.query)
        results = vs.retrieve(q, top_k=top_k)
        results = sorted(results, key=lambda x: x[0], reverse=True)

        reject = should_reject(results, abs_th=abs_th, gap_th=0.0)
        if reject == c.expect_reject:
            gate_correct += 1

        top1_score = results[0][0] if results else None
        if c.expect_reject:
            neg_total += 1
            if top1_score is not None:
                neg_top1_scores.append(float(top1_score))
        else:
            pos_total += 1
            if top1_score is not None:
                pos_top1_scores.append(float(top1_score))

            if c.must_contain:
                hit = any((c.must_contain in doc) for _, doc in results)
                if hit:
                    hit_at_k += 1

        if verbose:
            print("=" * 80)
            print(f"[{c.id}] {c.query}")
            print(f"- normalized: {q}")
            print(f"- expect_reject: {c.expect_reject}, actual_reject: {reject}")
            if top1_score is not None:
                print(f"- top1_score: {top1_score:.4f}")
            if c.must_contain:
                print(f"- must_contain: {c.must_contain}")
            if c.note:
                print(f"- note: {c.note}")
            print("- top results:")
            for i, (s, d) in enumerate(results[: min(5, len(results))], 1):
                preview = d.replace("\n", " ")
                print(f"  {i}. {s:.4f} | {preview[:120]}")

    def _mean(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        return sum(xs) / len(xs)

    report: Dict[str, Any] = {
        "total": total,
        "pos_total": pos_total,
        "neg_total": neg_total,
        "gate_accuracy": (gate_correct / total) if total else None,
        "hit_at_k": (hit_at_k / pos_total) if pos_total else None,
        "abs_threshold": abs_th,
        "top_k": top_k,
        "pos_top1_mean": _mean(pos_top1_scores),
        "neg_top1_mean": _mean(neg_top1_scores),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Mini LLM App - retrieval & gating evaluation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--qa_path", type=str, default=os.path.join("eval", "qa.jsonl"))
    parser.add_argument("--storage_dir", type=str, default=os.path.join("storage", "eval_vector"))
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--abs_th", type=float, default=0.42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.qa_path):
        raise FileNotFoundError(f"qa file not found: {args.qa_path}")
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data dir not found: {args.data_dir}")

    cases = load_qa_jsonl(args.qa_path)
    chunks = build_kb_chunks(args.data_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise RuntimeError("KB is empty, please put .md/.txt into data/")

    # Use a separate storage dir to avoid interfering with the app's runtime artifacts.
    vs = VectorStore(chunks, storage_dir=args.storage_dir, force_rebuild=True, log_prefix="[EvalVectorStore]")
    report = evaluate(vs, cases, top_k=args.top_k, abs_th=args.abs_th, verbose=args.verbose)

    os.makedirs("eval", exist_ok=True)
    out_path = os.path.join("eval", "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
