"""
Phase 4 – RAG Pipeline (Enhanced): Hybrid Retrieval + Reranker + LLM Generator.

Architecture (upgraded):
  Query → QueryRewrite → MetadataFilter → BM25+Dense → RRF
       → Reranker → ParentExpand → ContextCompress → LLM → Answer

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
# PARENT CONTEXT EXPANDER (Cải tiến 4)
# ══════════════════════════════════════════════════════════

class ParentContextExpander:
    """
    When a child chunk is retrieved (e.g., Khoản 2 of Điều 5),
    pull sibling chunks from the same section to provide full context.
    """

    def __init__(self, chunks: list[dict], parent_map_path: Path = None):
        self.chunk_lookup = {c["id"]: c for c in chunks}
        self.parent_map = {}

        if parent_map_path is None:
            parent_map_path = PROCESSED_DIR / "parent_map.json"

        if parent_map_path.exists():
            with open(parent_map_path, "r", encoding="utf-8") as f:
                self.parent_map = json.load(f)
            print(f"  [INIT] ParentContextExpander: {len(self.parent_map)} entries")
        else:
            print("  [WARN] parent_map.json not found. Run phase2b_chunk_normalize.py first.")
            print("         ParentContextExpander will be disabled.")

    def expand(
        self, top_chunks: list[dict], max_extra: int = 2
    ) -> list[dict]:
        """
        For each retrieved chunk, add up to max_extra sibling chunks
        from the same section. Avoids duplicates.
        """
        if not self.parent_map:
            return top_chunks

        seen_ids = {r["chunk"]["id"] for r in top_chunks}
        extra = []

        for r in top_chunks:
            chunk_id = r["chunk"]["id"]
            info = self.parent_map.get(chunk_id, {})
            siblings = info.get("section_siblings", [])

            added = 0
            for sib_id in siblings:
                if sib_id in seen_ids or added >= max_extra:
                    break
                if sib_id in self.chunk_lookup:
                    extra.append({
                        "chunk": self.chunk_lookup[sib_id],
                        "score": r["score"] * 0.5,
                        "rerank_score": r.get("rerank_score", 0) * 0.5,
                        "method": "parent_expand"
                    })
                    seen_ids.add(sib_id)
                    added += 1

        return top_chunks + extra


# ══════════════════════════════════════════════════════════
# CONTEXTUAL COMPRESSOR (Cải tiến 5)
# ══════════════════════════════════════════════════════════

class ContextualCompressor:
    """
    After reranking, extract only query-relevant sentences from each chunk.
    Uses keyword overlap scoring — no additional API call.
    """

    def compress(
        self, query: str, chunks: list[dict], max_sents_per_chunk: int = 5
    ) -> str:
        """
        Build compressed context string from retrieved chunks.
        Keeps section headers and top-scoring sentences per chunk.
        """
        query_tokens = set(re.findall(r'\w+', query.lower()))
        context_parts = []

        for r in chunks:
            chunk = r["chunk"]
            header_parts = [f"[{chunk.get('source', '')}]"]
            if chunk.get("chapter"):
                header_parts.append(chunk["chapter"])
            if chunk.get("section"):
                header_parts.append(chunk["section"])
            header = " - ".join(header_parts)

            # Score each sentence by keyword overlap with query
            sentences = re.split(r'(?<=[.;])\s+', chunk["text"])
            scored = []
            for sent in sentences:
                if len(sent.strip()) < 15:
                    continue
                sent_tokens = set(re.findall(r'\w+', sent.lower()))
                overlap = len(query_tokens & sent_tokens)
                # Boost sentences containing Điều/Khoản references
                if re.search(r'[Đđ]iều\s+\d+|[Kk]hoản\s+\d+', sent):
                    overlap += 2
                scored.append((overlap, sent))

            # Sort by relevance, take top N
            scored.sort(key=lambda x: x[0], reverse=True)
            top_sents = [s for _, s in scored[:max_sents_per_chunk]]

            if top_sents:
                context_parts.append(f"{header}\n" + " ".join(top_sents))

        return "\n\n".join(context_parts)


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
                max_new_tokens=256,
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

        # --- Enhancement components (new) ---
        self.metadata_filter = MetadataFilter(self.chunks)
        self.query_rewriter = QueryRewriter()
        self.parent_expander = ParentContextExpander(self.chunks)
        self.compressor = ContextualCompressor()

        print("\n[OK] RAG Pipeline (Enhanced) san sang!")

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

            # Step 3: Hybrid retrieval with multiple query variants
            all_candidates = []
            for variant in query_variants:
                dense_results = self.dense.search(
                    variant, top_k=TOP_K_RETRIEVAL
                )
                sparse_results = sparse_filtered.search(
                    variant, top_k=TOP_K_RETRIEVAL
                )
                # Apply metadata filter to dense results (post-filter)
                if len(filtered_chunks) < len(self.chunks):
                    filtered_sources = {
                        c.get("source", "") for c in filtered_chunks
                    }
                    dense_results = [
                        r for r in dense_results
                        if r["chunk"].get("source", "") in filtered_sources
                    ]
                all_candidates.extend(dense_results)
                all_candidates.extend(sparse_results)

            # RRF fusion across all variants
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

            # Step 5: Parent Context Expansion (Cải tiến 4)
            expanded = self.parent_expander.expand(top_chunks, max_extra=2)

            # Step 6: Contextual Compression (Cải tiến 5)
            context = self.compressor.compress(query, expanded)

            # Step 7: Generate
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
                "mode": "RAG-Enhanced"
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
