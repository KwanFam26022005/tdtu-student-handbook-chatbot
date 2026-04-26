"""
Phase 4 – RAG Pipeline: Hybrid Retrieval + Cross-Encoder Reranker + LLM Generator.

Architecture:
  Query → BM25 + Dense (bge-m3) → RRF Fusion → Reranker → Prompt → LLM → Answer

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
import numpy as np
from pathlib import Path
from typing import Optional

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
        
        print("  🔧 Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        self.chunks = chunks
        
        print(f"  🔧 Loading embedding model: {embed_model_name}...")
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
        
        print("  🔧 Building BM25 index...")
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
# RERANKER
# ══════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Reranker sử dụng Cross-Encoder (bge-reranker)"""
    
    def __init__(self, model_name=RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        
        print(f"  🔧 Loading reranker: {model_name}...")
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"  🔧 Loading LLM: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
        )
        
        # Load LoRA adapter nếu có
        if lora_path and Path(lora_path).exists():
            from peft import PeftModel
            print(f"  🔧 Loading LoRA adapter: {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.is_finetuned = True
        else:
            self.is_finetuned = False
        
        self.model.eval()
        print(f"  ✅ LLM ready! (Fine-tuned: {self.is_finetuned})")
    
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
    """Pipeline RAG hoàn chỉnh"""
    
    def __init__(self, use_finetuned=False):
        print("🚀 Khởi tạo RAG Pipeline...")
        print("=" * 60)
        
        # Load chunks
        chunks_path = PROCESSED_DIR / "chunks.json"
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"  📄 Loaded {len(self.chunks)} chunks")
        
        # Dense retriever
        faiss_path = PROCESSED_DIR / "faiss_index.bin"
        self.dense = DenseRetriever(faiss_path, self.chunks)
        
        # Sparse retriever
        self.sparse = SparseRetriever(self.chunks)
        
        # Hybrid
        self.hybrid = HybridRetriever(self.dense, self.sparse)
        
        # Reranker
        self.reranker = CrossEncoderReranker()
        
        # LLM
        lora_path = str(BASE_DIR / "outputs" / "finetune" / "lora_adapter") if use_finetuned else None
        self.llm = LLMGenerator(lora_path=lora_path)
        
        print("\n✅ RAG Pipeline sẵn sàng!")
    
    def answer(self, query: str, use_rag=True) -> dict:
        """
        Trả lời câu hỏi.
        
        Returns:
            dict với keys: answer, sources, retrieval_scores
        """
        if use_rag:
            # 1. Hybrid retrieval
            candidates = self.hybrid.search(query, top_k=TOP_K_RETRIEVAL)
            
            # 2. Rerank
            top_chunks = self.reranker.rerank(query, candidates, top_k=TOP_K_RERANK)
            
            # 3. Build context
            context = "\n\n".join([
                r["chunk"]["text_with_context"] for r in top_chunks
            ])
            
            # 4. Generate
            answer = self.llm.generate(query, context=context)
            
            return {
                "answer": answer,
                "sources": [r["chunk"]["source"] for r in top_chunks],
                "sections": [r["chunk"].get("section", "") for r in top_chunks],
                "retrieval_scores": [r["rerank_score"] for r in top_chunks],
                "mode": "RAG"
            }
        else:
            # No RAG: chỉ dùng LLM
            answer = self.llm.generate(query)
            return {
                "answer": answer,
                "sources": [],
                "sections": [],
                "retrieval_scores": [],
                "mode": "No-RAG"
            }


# ══════════════════════════════════════════════════════════
# TEST INTERACTIVE
# ══════════════════════════════════════════════════════════

def interactive_test():
    """Test pipeline với input thủ công"""
    pipeline = RAGPipeline(use_finetuned=False)
    
    print("\n" + "=" * 60)
    print("💬 CHATBOT SỔ TAY SINH VIÊN TDTU")
    print("   Gõ 'quit' để thoát")
    print("=" * 60)
    
    while True:
        query = input("\n🎓 Sinh viên hỏi: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        result = pipeline.answer(query, use_rag=True)
        
        print(f"\n🤖 Trợ lý: {result['answer']}")
        print(f"\n📚 Nguồn: {', '.join(result['sources'][:3])}")
        if result['sections']:
            print(f"📋 Phần: {', '.join(s for s in result['sections'][:3] if s)}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 4 – RAG Pipeline (Hybrid Retrieval + LLM)      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    interactive_test()
