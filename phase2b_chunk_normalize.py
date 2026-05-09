"""
Phase 2B – Chunk Normalization & Parent-Child Mapping.

Post-processes chunks.json from Phase 2:
  1. Merge tiny chunks (< 80 chars) into neighbors (same source only)
  2. Split oversized chunks (> 3000 chars) by paragraph boundary
  3. Build parent-child mapping for hierarchical retrieval
  4. Rebuild FAISS index with normalized chunks

Run AFTER phase2_process.py, BEFORE phase4_rag.py.

Usage:
  python phase2b_chunk_normalize.py
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
PROCESSED_DIR = BASE_DIR / "processed"

MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 3000
EMBEDDING_MODEL = "BAAI/bge-m3"


# ══════════════════════════════════════════════════════════
# STEP 1: MERGE TINY CHUNKS
# ══════════════════════════════════════════════════════════

def merge_tiny_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge chunks shorter than MIN_CHUNK_CHARS into their nearest neighbor.
    Only merges within the same source file to avoid cross-document contamination.
    """
    if not chunks:
        return chunks

    # Group by source to avoid cross-document merging
    from itertools import groupby
    groups = []
    for key, group in groupby(chunks, key=lambda c: c.get("source", "")):
        groups.append((key, list(group)))

    merged_all = []
    merge_count = 0

    for source, group_chunks in groups:
        merged = []
        buffer = None

        for chunk in group_chunks:
            if len(chunk["text"]) < MIN_CHUNK_CHARS:
                merge_count += 1
                if buffer is None:
                    buffer = chunk.copy()
                else:
                    buffer["text"] += "\n" + chunk["text"]
                    buffer["text_with_context"] += "\n" + chunk["text"]
            else:
                if buffer is not None:
                    # Prepend buffer to current chunk
                    chunk = chunk.copy()
                    chunk["text"] = buffer["text"] + "\n" + chunk["text"]
                    chunk["text_with_context"] = (
                        buffer["text_with_context"] + "\n" + chunk["text"]
                    )
                    buffer = None
                merged.append(chunk)

        # Handle trailing buffer
        if buffer is not None:
            if merged:
                last = merged[-1].copy()
                last["text"] += "\n" + buffer["text"]
                last["text_with_context"] += "\n" + buffer["text_with_context"]
                merged[-1] = last
            else:
                merged.append(buffer)

        merged_all.extend(merged)

    print(f"  Merged {merge_count} tiny chunks")
    return merged_all


# ══════════════════════════════════════════════════════════
# STEP 2: SPLIT OVERSIZED CHUNKS
# ══════════════════════════════════════════════════════════

def split_large_chunks(chunks: list[dict]) -> list[dict]:
    """
    Split chunks longer than MAX_CHUNK_CHARS at paragraph boundaries.
    Each sub-chunk inherits metadata and gets a parent_id link.
    """
    result = []
    split_count = 0

    for chunk in chunks:
        if len(chunk["text"]) <= MAX_CHUNK_CHARS:
            result.append(chunk)
            continue

        split_count += 1
        paragraphs = chunk["text"].split("\n\n")
        sub_texts = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > MAX_CHUNK_CHARS and current.strip():
                sub_texts.append(current.strip())
                current = para + "\n\n"
            else:
                current += para + "\n\n"

        if current.strip():
            sub_texts.append(current.strip())

        # Build context header once
        header_parts = [f"[{chunk['source']}]"]
        if chunk.get("chapter"):
            header_parts.append(chunk["chapter"])
        if chunk.get("section"):
            header_parts.append(chunk["section"])
        context_header = " - ".join(header_parts)

        for i, sub_text in enumerate(sub_texts):
            new_chunk = chunk.copy()
            new_chunk["text"] = sub_text
            new_chunk["text_with_context"] = f"{context_header}\n{sub_text}"
            new_chunk["parent_id"] = chunk["id"]
            new_chunk["sub_index"] = i
            result.append(new_chunk)

    print(f"  Split {split_count} oversized chunks")
    return result


# ══════════════════════════════════════════════════════════
# STEP 3: BUILD PARENT-CHILD MAP
# ══════════════════════════════════════════════════════════

def build_parent_map(chunks: list[dict]) -> dict:
    """
    Build a mapping: chunk_id -> {chapter, section, source, siblings}.
    Used by ParentContextExpander in Phase 4 to pull related chunks.
    """
    # Group chunks by (source, chapter)
    chapter_groups = defaultdict(list)
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("chapter", ""))
        chapter_groups[key].append(chunk["id"])

    # Group chunks by (source, section) for closer siblings
    section_groups = defaultdict(list)
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("section", ""))
        section_groups[key].append(chunk["id"])

    parent_map = {}
    for chunk in chunks:
        ch_key = (chunk.get("source", ""), chunk.get("chapter", ""))
        sec_key = (chunk.get("source", ""), chunk.get("section", ""))

        parent_map[chunk["id"]] = {
            "chapter": chunk.get("chapter", ""),
            "section": chunk.get("section", ""),
            "source": chunk.get("source", ""),
            "parent_id": chunk.get("parent_id", ""),
            "section_siblings": [
                cid for cid in section_groups.get(sec_key, [])
                if cid != chunk["id"]
            ],
            "chapter_siblings": [
                cid for cid in chapter_groups.get(ch_key, [])
                if cid != chunk["id"]
            ],
        }

    return parent_map


# ══════════════════════════════════════════════════════════
# STEP 4: REBUILD FAISS INDEX
# ══════════════════════════════════════════════════════════

def rebuild_faiss_index(chunks: list[dict]):
    """Re-embed normalized chunks and rebuild FAISS index."""
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text_with_context"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss_path = PROCESSED_DIR / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    print(f"  [OK] FAISS rebuilt: {faiss_path} (dim={dim}, n={index.ntotal})")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("+" + "=" * 58 + "+")
    print("|  Phase 2B – Chunk Normalization & Parent-Child Mapping  |")
    print("+" + "=" * 58 + "+\n")

    # Load original chunks
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        print(f"[ERROR] {chunks_path} not found. Run phase2_process.py first!")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    orig_count = len(chunks)
    orig_lens = [len(c["text"]) for c in chunks]
    print(f"[LOAD] {orig_count} chunks")
    print(f"  min={min(orig_lens)}, max={max(orig_lens)}, "
          f"avg={sum(orig_lens) // len(orig_lens)}")
    print(f"  Tiny (<{MIN_CHUNK_CHARS}): "
          f"{sum(1 for l in orig_lens if l < MIN_CHUNK_CHARS)}")
    print(f"  Oversized (>{MAX_CHUNK_CHARS}): "
          f"{sum(1 for l in orig_lens if l > MAX_CHUNK_CHARS)}")

    # Step 1: Merge tiny
    print(f"\n[STEP 1] Merging tiny chunks (<{MIN_CHUNK_CHARS} chars)...")
    chunks = merge_tiny_chunks(chunks)
    print(f"  Result: {len(chunks)} chunks")

    # Step 2: Split oversized
    print(f"\n[STEP 2] Splitting oversized chunks (>{MAX_CHUNK_CHARS} chars)...")
    chunks = split_large_chunks(chunks)
    print(f"  Result: {len(chunks)} chunks")

    # Re-assign sequential IDs
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"chunk_{i:04d}"

    # Step 3: Parent map
    print("\n[STEP 3] Building parent-child mapping...")
    parent_map = build_parent_map(chunks)

    # Save chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    parent_map_path = PROCESSED_DIR / "parent_map.json"
    with open(parent_map_path, "w", encoding="utf-8") as f:
        json.dump(parent_map, f, ensure_ascii=False, indent=2)

    metadata = [
        {"id": c["id"], "source": c["source"],
         "section": c.get("section", ""), "chapter": c.get("chapter", "")}
        for c in chunks
    ]
    meta_path = PROCESSED_DIR / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Stats
    new_lens = [len(c["text"]) for c in chunks]
    print(f"\n[STATS] Normalization Results:")
    print(f"  Before: {orig_count} chunks | "
          f"min={min(orig_lens)} max={max(orig_lens)}")
    print(f"  After:  {len(chunks)} chunks | "
          f"min={min(new_lens)} max={max(new_lens)}")
    print(f"  Avg: {sum(new_lens) // len(new_lens)} chars")

    # Step 4: Rebuild FAISS
    print("\n[STEP 4] Rebuilding FAISS index...")
    rebuild_faiss_index(chunks)

    print(f"\n[OK] Phase 2B complete!")
    print(f"  chunks.json:   {chunks_path}")
    print(f"  parent_map:    {parent_map_path}")
    print(f"  FAISS index:   {PROCESSED_DIR / 'faiss_index.bin'}")


if __name__ == "__main__":
    main()
