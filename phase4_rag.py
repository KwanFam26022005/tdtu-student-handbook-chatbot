"""
Phase 4 – RAG Pipeline (Enhanced): Hybrid Retrieval + Reranker + LLM Generator.

Architecture (upgraded):
  Query → QueryRewrite → HyDE → MetadataFilter → BM25+Dense → RRF
       → Reranker → HierarchicalExpand → SemanticCompress
       → CRAGGate → LLM → Answer

Components:
  - Dense retriever: FAISS + bge-m3
  - Sparse retriever: BM25 (rank_bm25)
  - Fusion: Reciprocal Rank Fusion (RRF)
  - Reranker: bge-reranker-v2-m3
  - Generator: Qwen2.5-3B (base hoặc fine-tuned)

Cài đặt:
  pip install sentence-transformers faiss-cpu rank-bm25 transformers torch
  
Chạy:
  python phase4_rag.py
"""

import json
import re
import os
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
PROCESSED_DIR = BASE_DIR / "processed"

# Retrieval config
TOP_K_RETRIEVAL = 20   # Số chunks lấy từ mỗi retriever
TOP_K_RERANK = 5       # Số chunks sau rerank (đưa vào LLM)
TOP_K_FINAL = 5        # Recall@5

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_BASE = "Qwen/Qwen2.5-3B-Instruct"

# System prompt
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về quy chế và sổ tay sinh viên Trường Đại học Tôn Đức Thắng (TDTU).
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên một cách chính xác, rõ ràng và thân thiện.
Hãy trả lời dựa HOÀN TOÀN vào ngữ cảnh được cung cấp bên dưới.
Khi trả lời, hãy trích dẫn cụ thể tên văn bản, số điều, khoản trong quy chế.
Nếu ngữ cảnh không chứa thông tin liên quan, hãy nói rõ "Không tìm thấy thông tin trong quy chế hiện có"."""


# ══════════════════════════════════════════════════════════
# RETRIEVER COMPONENTS
# ══════════════════════════════════════════════════════════

class DenseRetriever:
    """Dense retriever sử dụng FAISS + bge-m3"""
    
    def __init__(self, index_path, chunks, embed_model_name=EMBEDDING_MODEL):
        import faiss
        from sentence_transformers import SentenceTransformer
        
        print("  [INIT] Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        self.chunks = chunks
        
        print(f"  [INIT] Loading embedding model: {embed_model_name}...")
        self.embed_model = SentenceTransformer(embed_model_name)
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Tìm top-k chunks gần nhất theo cosine similarity"""
        query_emb = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype('float32')
        
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "method": "dense"
                })
        return results


class SparseRetriever:
    """Sparse retriever sử dụng BM25"""
    
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        
        print("  [INIT] Building BM25 index...")
        self.chunks = chunks
        
        # Tokenize đơn giản cho tiếng Việt
        self.tokenized_corpus = [
            self._tokenize(c["text_with_context"]) for c in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer (đủ cho BM25)"""
        import re
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Tìm top-k chunks theo BM25 score"""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "method": "sparse"
                })
        return results


class HybridRetriever:
    """Kết hợp Dense + Sparse bằng Reciprocal Rank Fusion (RRF)"""
    
    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Hybrid search với RRF fusion"""
        dense_results = self.dense.search(query, top_k)
        sparse_results = self.sparse.search(query, top_k)
        
        # RRF: score = Σ 1/(k + rank)  với k=60 (constant)
        k = 60
        rrf_scores = {}
        
        for rank, r in enumerate(dense_results):
            chunk_id = r["chunk"]["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        
        for rank, r in enumerate(sparse_results):
            chunk_id = r["chunk"]["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        
        # Build chunk lookup
        chunk_lookup = {}
        for r in dense_results + sparse_results:
            chunk_lookup[r["chunk"]["id"]] = r["chunk"]
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for chunk_id in sorted_ids[:top_k]:
            results.append({
                "chunk": chunk_lookup[chunk_id],
                "score": rrf_scores[chunk_id],
                "method": "hybrid_rrf"
            })
        
        return results


# ══════════════════════════════════════════════════════════
# METADATA FILTER (Cải tiến 2)
# ══════════════════════════════════════════════════════════

class MetadataFilter:
    """
    Pre-filter chunks by source/chapter based on query keywords.
    Auto-builds keyword→source mapping from chunk content.
    Falls back to full set if filtered results are too few.
    """

    # Query topic patterns → domain keywords in chunk text
    TOPIC_PATTERNS = {
        'tot_nghiep': r'tốt nghiệp|ra trường|nhận bằng|xét tốt nghiệp|bằng tốt nghiệp',
        'hoc_phi': r'học phí|đóng tiền|miễn giảm|học bổng|trả tiền|công nợ',
        'ky_luat': r'khen thưởng|kỷ luật|vi phạm|cảnh cáo|đình chỉ|buộc thôi|đuổi học',
        'dao_tao': r'đào tạo|tín chỉ|học kỳ|đăng ký|môn học|điểm số|GPA|CPA|xét học vụ',
        'ren_luyen': r'rèn luyện|điểm rèn luyện|hạnh kiểm|đánh giá rèn luyện',
        'thuc_tap': r'thực tập|khóa luận|đồ án|luận văn|tốt nghiệp',
        'ky_tuc_xa': r'ký túc|nội trú|phòng ở|chỗ ở|khu nội trú',
        'thu_vien': r'thư viện|mượn sách|tài liệu|phòng đọc',
    }

    def __init__(self, chunks: list[dict]):
        self.source_topics = defaultdict(set)  # topic → set of source names
        self._build_index(chunks)

    def _build_index(self, chunks: list[dict]):
        """Scan all chunks to map topics → sources."""
        for chunk in chunks:
            source = chunk.get("source", "")
            text_lower = chunk["text"].lower()
            for topic, pattern in self.TOPIC_PATTERNS.items():
                if re.search(pattern, text_lower):
                    self.source_topics[topic].add(source)
        print(f"  [INIT] MetadataFilter: {len(self.source_topics)} topics indexed")

    def filter_chunks(
        self, query: str, chunks: list[dict], min_results: int = 20
    ) -> list[dict]:
        """
        Filter chunks to only include relevant sources.
        Returns full set if no topic matches or too few results.
        """
        query_lower = query.lower()
        relevant_sources = set()

        for topic, pattern in self.TOPIC_PATTERNS.items():
            if re.search(pattern, query_lower):
                relevant_sources.update(self.source_topics.get(topic, set()))

        if not relevant_sources:
            return chunks  # No match → no filter

        filtered = [
            c for c in chunks if c.get("source", "") in relevant_sources
        ]

        if len(filtered) < min_results:
            return chunks  # Too few → fallback

        return filtered


# ══════════════════════════════════════════════════════════
# QUERY REWRITER (Cải tiến 3)
# ══════════════════════════════════════════════════════════

class QueryRewriter:
    """
    Rewrite informal student queries into formal regulatory language.
    Uses dictionary-based synonym mapping (no API call needed).
    """

    SYNONYM_MAP = {
        'đuổi học': 'buộc thôi học',
        'bị đuổi': 'bị buộc thôi học',
        'nghỉ học': 'bảo lưu kết quả học tập',
        'bỏ học': 'thôi học tự nguyện',
        'thi lại': 'thi kết thúc học phần lần hai',
        'học lại': 'đăng ký học lại học phần',
        'trượt': 'không đạt',
        'rớt': 'không đạt',
        'rớt môn': 'không đạt học phần',
        'điểm liệt': 'điểm dưới mức tối thiểu',
        'nợ môn': 'chưa hoàn thành học phần',
        'GPA': 'điểm trung bình tích lũy',
        'ra trường': 'xét tốt nghiệp',
        'deadline': 'thời hạn quy định',
        'chuyển trường': 'chuyển cơ sở đào tạo',
        'bị cảnh cáo': 'cảnh báo học vụ',
        'xin phép': 'đơn đề nghị',
        'đóng tiền': 'nộp học phí',
        'hoãn thi': 'tạm hoãn thi kết thúc học phần',
        'gap year': 'bảo lưu kết quả học tập',
        'chuyển ngành': 'chuyển ngành đào tạo',
        'điểm danh': 'theo dõi chuyên cần',
        'vắng thi': 'vắng mặt trong kỳ thi',
        'phúc khảo': 'phúc tra bài thi',
    }

    def rewrite(self, query: str) -> list[str]:
        """
        Return query variants: [original, synonym-rewritten, keyword-extracted].
        Multiple variants are searched and results merged via RRF.
        """
        variants = [query]

        # Dictionary-based rewrite
        rewritten = query
        for informal, formal in self.SYNONYM_MAP.items():
            if informal.lower() in query.lower():
                rewritten = re.sub(
                    re.escape(informal), formal, rewritten,
                    flags=re.IGNORECASE
                )

        if rewritten != query:
            variants.append(rewritten)

        # Extract Điều/Khoản numbers for targeted search
        dieu = re.search(r'[Đđ]iều\s+(\d+)', query)
        if dieu:
            variants.append(f"Điều {dieu.group(1)}")

        khoan = re.search(r'[Kk]hoản\s+(\d+)', query)
        if khoan and dieu:
            variants.append(f"Điều {dieu.group(1)} Khoản {khoan.group(1)}")

        return variants


# ══════════════════════════════════════════════════════════
# HyDE — HYPOTHETICAL DOCUMENT EMBEDDING (Cải tiến 6)
# ══════════════════════════════════════════════════════════

class HyDEGenerator:
    """
    Hypothetical Document Embedding — Gao et al. (2022),
    "Precise Zero-Shot Dense Retrieval without Relevance Labels".

    Thay vì embed câu hỏi gốc của sinh viên, LLM sinh ra một đoạn
    quy chế GIẢ ĐỊNH có thể trả lời câu hỏi đó. Embedding của đoạn
    giả định gần với phong cách văn bản pháp lý hơn → cải thiện
    độ chính xác của dense retrieval.

    Ví dụ:
        Query     : "nếu điểm chuyên cần dưới 50% thì có bị cảnh báo không?"
        Hypothesis: "Sinh viên có điểm chuyên cần dưới 50% trong học kỳ
                     sẽ bị cảnh báo học vụ theo Điều 16 Khoản 1..."
    """

    HYDE_SYSTEM = (
        "Bạn là chuyên gia về quy chế TDTU. "
        "Hãy viết một đoạn quy chế NGẮN (2-4 câu, dưới 100 từ) "
        "giống phong cách văn bản pháp lý, có thể trả lời câu hỏi sau. "
        "Chỉ viết đoạn quy chế, không giải thích thêm."
    )

    def __init__(self, llm_generator):
        """Nhận LLMGenerator đã khởi tạo để tái sử dụng model/tokenizer."""
        self.llm = llm_generator

    def generate_hypothesis(self, query: str) -> str:
        """
        Sinh một đoạn quy chế giả định từ câu hỏi.
        Dùng greedy decoding (do_sample=False) để ổn định, tối đa 120 tokens.
        """
        messages = [
            {"role": "system", "content": self.HYDE_SYSTEM},
            {"role": "user", "content": query},
        ]
        text = self.llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.llm.tokenizer(text, return_tensors="pt").to(
            self.llm.model.device
        )

        import torch
        with torch.no_grad():
            outputs = self.llm.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
            )
        hypothesis = self.llm.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        print(f"  [HyDE] Hypothesis: {hypothesis[:80]}...")
        return hypothesis


# ══════════════════════════════════════════════════════════
# HIERARCHICAL CONTEXT EXPANDER (Cải tiến 4 — nâng cấp)
# ══════════════════════════════════════════════════════════

class HierarchicalExpander:
    """
    Khai thác cấu trúc cây pháp lý: Chương > Điều > Khoản > điểm a/b/c.

    Khi retrieve được 1 chunk thuộc Khoản 2 Điều 5, tự động:
      1. Kéo toàn bộ Khoản cùng Điều (sibling)
      2. Kéo header Điều (parent) nếu chunk không chứa header
      3. Ưu tiên Khoản liền kề (Khoản 1, 3) trước Khoản xa
    """

    def __init__(self, chunks: list[dict], parent_map_path: Path = None):
        self.chunk_lookup = {c["id"]: c for c in chunks}
        self.parent_map = {}

        # Build indexes: section → [chunk_ids], chapter → [chunk_ids]
        self.section_index: dict[str, list[str]] = defaultdict(list)  # "Điều 5" → [id1, id2...]
        self.chapter_index: dict[str, list[str]] = defaultdict(list)

        for c in chunks:
            sec = c.get("section", "")
            chap = c.get("chapter", "")
            src = c.get("source", "")
            if sec:
                self.section_index[f"{src}||{sec}"].append(c["id"])
            if chap:
                self.chapter_index[f"{src}||{chap}"].append(c["id"])

        if parent_map_path is None:
            parent_map_path = PROCESSED_DIR / "parent_map.json"
        if parent_map_path.exists():
            with open(parent_map_path, "r", encoding="utf-8") as f:
                self.parent_map = json.load(f)

        print(f"  [INIT] HierarchicalExpander: {len(self.section_index)} sections, "
              f"{len(self.chapter_index)} chapters indexed")

    def expand(
        self, top_chunks: list[dict], max_extra: int = 2
    ) -> list[dict]:
        """
        Expand retrieved chunks theo cấu trúc cây pháp lý.
        Ưu tiên: cùng Điều > cùng Chương > parent_map siblings.
        """
        seen_ids = {r["chunk"]["id"] for r in top_chunks}
        extra = []

        for r in top_chunks:
            chunk = r["chunk"]
            chunk_id = chunk["id"]
            src = chunk.get("source", "")
            sec = chunk.get("section", "")

            # Ưu tiên 1: Kéo siblings cùng Điều (cùng source)
            added = 0
            if sec:
                section_key = f"{src}||{sec}"
                for sib_id in self.section_index.get(section_key, []):
                    if sib_id in seen_ids or added >= max_extra:
                        break
                    extra.append({
                        "chunk": self.chunk_lookup[sib_id],
                        "score": r["score"] * 0.6,
                        "rerank_score": r.get("rerank_score", 0) * 0.6,
                        "method": "hierarchical_section"
                    })
                    seen_ids.add(sib_id)
                    added += 1

            # Ưu tiên 2: Fallback sang parent_map siblings (nếu còn slot)
            if added < max_extra and chunk_id in self.parent_map:
                siblings = self.parent_map[chunk_id].get("section_siblings", [])
                for sib_id in siblings:
                    if sib_id in seen_ids or added >= max_extra:
                        break
                    if sib_id in self.chunk_lookup:
                        extra.append({
                            "chunk": self.chunk_lookup[sib_id],
                            "score": r["score"] * 0.4,
                            "rerank_score": r.get("rerank_score", 0) * 0.4,
                            "method": "hierarchical_parent"
                        })
                        seen_ids.add(sib_id)
                        added += 1

        return top_chunks + extra


# ══════════════════════════════════════════════════════════
# SEMANTIC COMPRESSOR (Cải tiến 5 — nâng cấp)
# ══════════════════════════════════════════════════════════

class SemanticCompressor:
    """
    Thay thế keyword overlap bằng sentence embedding similarity.
    Dùng bge-m3 đã load sẵn từ DenseRetriever → không tốn thêm VRAM.

    So với ContextualCompressor cũ (keyword overlap):
      - Hiểu ngữ nghĩa: "buộc thôi học" ≈ "đuổi học" (overlap=0 nhưng cosine≈0.8)
      - Giữ câu pháp lý quan trọng: "trừ trường hợp quy định tại khoản 2"
      - Vẫn giữ boost cho Điều/Khoản references
    """

    def __init__(self, embed_model=None):
        """
        Args:
            embed_model: SentenceTransformer instance (tái sử dụng từ DenseRetriever)
        """
        self.embed_model = embed_model

    def compress(
        self, query: str, chunks: list[dict], max_sents_per_chunk: int = 5
    ) -> str:
        """
        Build compressed context. Chọn câu theo cosine similarity với query,
        sau đó sắp xếp lại theo thứ tự gốc trong văn bản.
        """
        context_parts = []

        # Encode query một lần
        if self.embed_model:
            query_emb = self.embed_model.encode(
                [query], normalize_embeddings=True
            )
        else:
            query_emb = None

        for r in chunks:
            chunk = r["chunk"]
            header_parts = [f"[{chunk.get('source', '')}]"]
            if chunk.get("chapter"):
                header_parts.append(chunk["chapter"])
            if chunk.get("section"):
                header_parts.append(chunk["section"])
            header = " - ".join(header_parts)

            sentences = re.split(r'(?<=[.;])\s+', chunk["text"])
            valid_sents = [
                (i, s) for i, s in enumerate(sentences) if len(s.strip()) >= 15
            ]

            if not valid_sents:
                continue

            if query_emb is not None and len(valid_sents) > max_sents_per_chunk:
                # Semantic scoring: cosine similarity
                sent_texts = [s for _, s in valid_sents]
                sent_embs = self.embed_model.encode(
                    sent_texts, normalize_embeddings=True
                )
                similarities = (sent_embs @ query_emb.T).flatten()

                indexed_scored = []
                for j, (orig_idx, sent) in enumerate(valid_sents):
                    score = float(similarities[j])
                    # Boost sentences containing Điều/Khoản references
                    if re.search(r'[Đđ]iều\s+\d+|[Kk]hoản\s+\d+', sent):
                        score += 0.15
                    # Boost cross-reference sentences ("quy định tại...")
                    if re.search(r'quy định tại|theo quy định|trừ trường hợp', sent, re.IGNORECASE):
                        score += 0.1
                    indexed_scored.append((orig_idx, score, sent))

                # Chọn top-N theo score, rồi sắp xếp lại theo thứ tự gốc
                top_by_score = sorted(indexed_scored, key=lambda x: x[1], reverse=True)[:max_sents_per_chunk]
                top_by_order = sorted(top_by_score, key=lambda x: x[0])
                top_sents = [s for _, _, s in top_by_order]
            else:
                # Fallback: giữ nguyên (ít câu, hoặc không có embed model)
                top_sents = [s for _, s in valid_sents[:max_sents_per_chunk]]

            if top_sents:
                context_parts.append(f"{header}\n" + " ".join(top_sents))

        return "\n\n".join(context_parts)


# ══════════════════════════════════════════════════════════
# CRAG — CORRECTIVE RAG RELEVANCE GATE (Cải tiến 7)
# ══════════════════════════════════════════════════════════

class CRAGRelevanceGate:
    """
    Corrective RAG — Yan et al. (2024).

    Kiểm tra chất lượng retrieval trước khi đưa vào LLM:
      - Nếu reranker score cao → CORRECT: dùng context bình thường
      - Nếu score trung bình → AMBIGUOUS: thêm cảnh báo vào prompt
      - Nếu score thấp → INCORRECT: không dùng context, trả lời fallback

    Tránh LLM hallucinate dựa trên context noise.
    """

    def __init__(self, high_threshold: float = 0.5, low_threshold: float = -2.0):
        """
        Thresholds cho cross-encoder score (bge-reranker-v2-m3).
        Typical range: -10 → +10, với >0 thường là relevant.
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate(self, top_chunks: list[dict]) -> dict:
        """
        Đánh giá chất lượng retrieval.

        Returns:
            dict: {"verdict": "CORRECT"|"AMBIGUOUS"|"INCORRECT",
                   "top_score": float, "avg_score": float, "reason": str}
        """
        if not top_chunks:
            return {
                "verdict": "INCORRECT",
                "top_score": 0.0,
                "avg_score": 0.0,
                "reason": "Không tìm thấy chunks liên quan"
            }

        scores = [r.get("rerank_score", 0.0) for r in top_chunks]
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)

        if top_score >= self.high_threshold:
            return {
                "verdict": "CORRECT",
                "top_score": top_score,
                "avg_score": avg_score,
                "reason": f"Top reranker score {top_score:.2f} >= {self.high_threshold}"
            }
        elif top_score >= self.low_threshold:
            return {
                "verdict": "AMBIGUOUS",
                "top_score": top_score,
                "avg_score": avg_score,
                "reason": f"Score {top_score:.2f} ở vùng không chắc chắn"
            }
        else:
            return {
                "verdict": "INCORRECT",
                "top_score": top_score,
                "avg_score": avg_score,
                "reason": f"Top score {top_score:.2f} < {self.low_threshold} — context không liên quan"
            }


# ══════════════════════════════════════════════════════════
# RERANKER
# ══════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Reranker sử dụng Cross-Encoder (bge-reranker)"""
    
    def __init__(self, model_name=RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        
        print(f"  [INIT] Loading reranker: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
    
    def rerank(self, query: str, results: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
        """Rerank results bằng cross-encoder"""
        if not results:
            return []
        
        pairs = [(query, r["chunk"]["text_with_context"]) for r in results]
        scores = self.model.predict(pairs)
        
        # Gán score mới
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# ══════════════════════════════════════════════════════════
# LLM GENERATOR
# ══════════════════════════════════════════════════════════

class LLMGenerator:
    """Generate answer bằng LLM (base hoặc fine-tuned)"""
    
    def __init__(self, model_name=LLM_BASE, lora_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        print(f"  [INIT] Loading LLM: {model_name}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        # Load LoRA adapter nếu có
        if lora_path and Path(lora_path).exists():
            from peft import PeftModel
            print(f"  [INIT] Loading LoRA adapter: {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.is_finetuned = True
        else:
            self.is_finetuned = False
        
        self.model.eval()
        print(f"  [OK] LLM ready! (Fine-tuned: {self.is_finetuned})")
    
    def generate(self, query: str, context: str = "") -> str:
        """Generate answer cho query, với hoặc không có RAG context"""
        
        if context:
            # RAG mode: có context
            user_message = f"""Ngữ cảnh từ quy chế TDTU:
---
{context}
---

Câu hỏi: {query}
Hãy trả lời dựa trên ngữ cảnh trên."""
        else:
            # No-RAG mode: chỉ dùng kiến thức sẵn
            user_message = query
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )
        
        # Decode chỉ phần answer (bỏ prompt)
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return answer.strip()


# ══════════════════════════════════════════════════════════
# RAG PIPELINE (đầy đủ)
# ══════════════════════════════════════════════════════════

class RAGPipeline:
    """Pipeline RAG hoàn chỉnh (Enhanced with 4 improvements)"""

    def __init__(self, use_finetuned=False):
        print("[INIT] Khoi tao RAG Pipeline (Enhanced)...")
        print("=" * 60)

        # Load chunks
        chunks_path = PROCESSED_DIR / "chunks.json"
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"  [INFO] Loaded {len(self.chunks)} chunks")

        # --- Core components (unchanged) ---
        faiss_path = PROCESSED_DIR / "faiss_index.bin"
        self.dense = DenseRetriever(faiss_path, self.chunks)
        self.sparse = SparseRetriever(self.chunks)
        self.hybrid = HybridRetriever(self.dense, self.sparse)
        self.reranker = CrossEncoderReranker()

        lora_path = (
            str(BASE_DIR / "outputs" / "finetune" / "lora_adapter")
            if use_finetuned else None
        )
        self.llm = LLMGenerator(lora_path=lora_path)

        # --- Enhancement components ---
        self.metadata_filter = MetadataFilter(self.chunks)
        self.query_rewriter = QueryRewriter()
        self.hierarchical_expander = HierarchicalExpander(self.chunks)
        # SemanticCompressor tái sử dụng embed model từ DenseRetriever
        self.compressor = SemanticCompressor(embed_model=self.dense.embed_model)
        # HyDE dùng lại LLM đã load — không tốn thêm VRAM
        self.hyde = HyDEGenerator(self.llm)
        # CRAG relevance gate — kiểm tra chất lượng retrieval
        self.crag_gate = CRAGRelevanceGate()

        print("\n[OK] RAG Pipeline (Enhanced + HyDE + CRAG) san sang!")

    def cleanup(self):
        """Giải phóng GPU memory — gọi giữa các config trên Colab."""
        import torch
        import gc

        # Delete GPU-heavy components
        if hasattr(self, 'llm'):
            del self.llm
        if hasattr(self, 'reranker'):
            del self.reranker
        if hasattr(self, 'dense'):
            del self.dense

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CLEANUP] GPU memory freed.")

    def answer(self, query: str, use_rag=True) -> dict:
        """
        Trả lời câu hỏi — enhanced pipeline.

        Flow: Rewrite → Filter → Retrieve → Rerank → Expand → Compress → Generate

        Returns:
            dict với keys: answer, sources, retrieval_scores, mode, query_variants
        """
        if use_rag:
            # Step 1: Query Rewriting (Cải tiến 3)
            query_variants = self.query_rewriter.rewrite(query)

            # Step 2: Metadata Pre-filtering (Cải tiến 2)
            filtered_chunks = self.metadata_filter.filter_chunks(
                query, self.chunks
            )
            # Rebuild sparse retriever on filtered set if different
            if len(filtered_chunks) < len(self.chunks):
                sparse_filtered = SparseRetriever(filtered_chunks)
            else:
                sparse_filtered = self.sparse

            # Step 2.5: HyDE — sinh hypothesis để cải thiện dense retrieval (Cải tiến 6)
            hypothesis = self.hyde.generate_hypothesis(query)
            # Dense dùng thêm hypothesis; BM25 chỉ dùng query gốc (keyword match)
            dense_query_variants = query_variants + [hypothesis]

            # Pre-compute filtered_sources một lần duy nhất
            filtered_sources = (
                {c.get("source", "") for c in filtered_chunks}
                if len(filtered_chunks) < len(self.chunks) else None
            )

            # Step 3: Hybrid retrieval
            all_candidates = []

            # Dense retrieval: query_variants + hypothesis (HyDE)
            for variant in dense_query_variants:
                dense_results = self.dense.search(variant, top_k=TOP_K_RETRIEVAL)
                if filtered_sources:
                    dense_results = [
                        r for r in dense_results
                        if r["chunk"].get("source", "") in filtered_sources
                    ]
                all_candidates.extend(dense_results)

            # Sparse (BM25): chỉ query_variants gốc (hypothesis không tốt cho keyword match)
            for variant in query_variants:
                sparse_results = sparse_filtered.search(variant, top_k=TOP_K_RETRIEVAL)
                all_candidates.extend(sparse_results)

            # RRF fusion across all candidates
            k_rrf = 60
            rrf_scores = {}
            chunk_lookup = {}
            for rank, r in enumerate(all_candidates):
                cid = r["chunk"]["id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k_rrf + rank + 1)
                chunk_lookup[cid] = r["chunk"]

            sorted_ids = sorted(
                rrf_scores, key=rrf_scores.get, reverse=True
            )
            candidates = [
                {"chunk": chunk_lookup[cid], "score": rrf_scores[cid],
                 "method": "hybrid_rrf"}
                for cid in sorted_ids[:TOP_K_RETRIEVAL]
            ]

            # Step 4: Rerank
            top_chunks = self.reranker.rerank(
                query, candidates, top_k=TOP_K_RERANK
            )

            # Step 5: Hierarchical Context Expansion (Cải tiến 4 — nâng cấp)
            expanded = self.hierarchical_expander.expand(top_chunks, max_extra=2)

            # Step 6: CRAG Relevance Gate (Cải tiến 7)
            crag_result = self.crag_gate.evaluate(top_chunks)
            print(f"  [CRAG] Verdict: {crag_result['verdict']} "
                  f"(top={crag_result['top_score']:.2f}, avg={crag_result['avg_score']:.2f})")

            if crag_result["verdict"] == "INCORRECT":
                # Context không liên quan → trả lời fallback, không hallucinate
                return {
                    "answer": "Không tìm thấy thông tin liên quan trong quy chế hiện có. "
                              "Vui lòng liên hệ phòng Công tác Sinh viên để được hỗ trợ.",
                    "sources": [],
                    "sections": [],
                    "retrieval_scores": [r.get("rerank_score", 0) for r in top_chunks],
                    "query_variants": query_variants,
                    "hyde_hypothesis": hypothesis,
                    "crag": crag_result,
                    "mode": "RAG-CRAG-Rejected"
                }

            # Step 7: Semantic Compression (Cải tiến 5 — nâng cấp)
            context = self.compressor.compress(query, expanded)

            # Step 7.5: Nếu CRAG AMBIGUOUS → thêm cảnh báo vào prompt
            if crag_result["verdict"] == "AMBIGUOUS":
                context = ("[LƯU Ý: Thông tin dưới đây có thể không hoàn toàn chính xác "
                           "cho câu hỏi này. Hãy trả lời thận trọng và ghi rõ nếu không "
                           "chắc chắn.]\n\n" + context)

            # Step 8: Generate
            answer = self.llm.generate(query, context=context)

            return {
                "answer": answer,
                "sources": [r["chunk"]["source"] for r in top_chunks],
                "sections": [
                    r["chunk"].get("section", "") for r in top_chunks
                ],
                "retrieval_scores": [
                    r["rerank_score"] for r in top_chunks
                ],
                "query_variants": query_variants,
                "hyde_hypothesis": hypothesis,
                "crag": crag_result,
                "mode": "RAG-Enhanced+HyDE+CRAG"
            }
        else:
            # No RAG: chỉ dùng LLM
            answer = self.llm.generate(query)
            return {
                "answer": answer,
                "sources": [],
                "sections": [],
                "retrieval_scores": [],
                "query_variants": [],
                "mode": "No-RAG"
            }


# ══════════════════════════════════════════════════════════
# TEST INTERACTIVE
# ══════════════════════════════════════════════════════════

def interactive_test():
    """Test pipeline với input thủ công"""
    pipeline = RAGPipeline(use_finetuned=False)
    
    print("\n" + "=" * 60)
    print("CHATBOT SO TAY SINH VIEN TDTU")
    print("   Go 'quit' de thoat")
    print("=" * 60)
    
    while True:
        query = input("\nSinh vien hoi: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        result = pipeline.answer(query, use_rag=True)
        
        print(f"\n[ANSWER] {result['answer']}")
        print(f"\n[SOURCE] {', '.join(result['sources'][:3])}")
        if result['sections']:
            print(f"[SECTION] {', '.join(s for s in result['sections'][:3] if s)}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 4 – RAG Pipeline (Hybrid Retrieval + LLM)      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    interactive_test()
