"""
Phase 3 (Unsloth) – Fine-tuning Qwen2.5-3B-Instruct với QLoRA trên Colab.

Phiên bản sử dụng Unsloth để tăng tốc 2x và tự động xử lý dtype/precision.
Nếu Unsloth bị lỗi import, dùng phase3_finetune.py (bản thuần Hugging Face).

Cài đặt (trên Colab):
  !pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install --upgrade --no-cache-dir --no-deps unsloth_zoo
  !pip install -Uq --no-deps trl peft accelerate bitsandbytes

Chạy:
  Restart runtime sau khi cài xong, rồi: python phase3_finetune_unsloth.py
"""

import os
import json
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")

# Model config
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"  # Bản Unsloth (tối ưu sẵn)
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training config
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 10

# Output
OUTPUT_DIR_TRAIN = BASE_DIR / "outputs" / "finetune"
HF_REPO_ID = "KwanFam26022005/tdtu-qwen2.5-3b-lora"

# System prompt cho chatbot
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về quy chế và sổ tay sinh viên Trường Đại học Tôn Đức Thắng (TDTU).
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên một cách chính xác, rõ ràng và thân thiện.
Khi trả lời, hãy trích dẫn cụ thể số điều, khoản trong quy chế nếu có.
Nếu không chắc chắn, hãy nói rõ rằng bạn không tìm thấy thông tin."""


# ══════════════════════════════════════════════════════════
# 1. LOAD MODEL (Unsloth)
# ══════════════════════════════════════════════════════════

def load_model():
    """Load Qwen2.5-3B-Instruct với Unsloth (4-bit + LoRA, tự xử lý dtype)"""
    from unsloth import FastLanguageModel

    print("[INIT] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto detect
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("[INIT] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Tiet kiem 30% VRAM
        random_state=42,
    )

    # In thong tin model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model loaded!")
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ══════════════════════════════════════════════════════════
# 2. PREPARE DATASET
# ══════════════════════════════════════════════════════════

def prepare_dataset(tokenizer):
    """Chuyen QA pairs sang chat format cho SFTTrainer"""

    qa_train_path = BASE_DIR / "processed" / "qa_train.json"
    if not qa_train_path.exists():
        print(f"[ERROR] Khong tim thay {qa_train_path}")
        print("   Chay phase2_process.py truoc!")
        return None

    with open(qa_train_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"[INFO] Loaded {len(qa_pairs)} QA pairs")

    # Chuyen sang chat format
    formatted_data = []
    for qa in qa_pairs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_data.append({"text": text})

    # Tao HuggingFace Dataset
    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data)

    print(f"[OK] Dataset: {len(dataset)} samples")
    print(f"   Vi du: {dataset[0]['text'][:200]}...")

    return dataset


# ══════════════════════════════════════════════════════════
# 3. TRAIN
# ══════════════════════════════════════════════════════════

def train(model, tokenizer, dataset):
    """Fine-tune model bang SFTTrainer (Unsloth compatible)"""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    OUTPUT_DIR_TRAIN.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,  # Pack multiple samples vao 1 sequence
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(OUTPUT_DIR_TRAIN),
            save_strategy="epoch",
            report_to="none",
        ),
    )

    print("\n[TRAIN] Bat dau fine-tuning...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size (effective): {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"   Learning rate: {LEARNING_RATE}")

    # GPU memory truoc training
    if torch.cuda.is_available():
        print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

    # Train
    stats = trainer.train()

    print(f"\n[OK] Training hoan tat!")
    print(f"   Total steps: {stats.global_step}")
    print(f"   Training loss: {stats.training_loss:.4f}")

    return trainer


# ══════════════════════════════════════════════════════════
# 4. SAVE & PUSH
# ══════════════════════════════════════════════════════════

def save_model(model, tokenizer):
    """Luu LoRA adapter + merge model + push len HuggingFace Hub"""

    # Luu local
    lora_path = OUTPUT_DIR_TRAIN / "lora_adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"[OK] LoRA adapter saved: {lora_path}")

    # Push len HuggingFace Hub (optional)
    try:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            model.push_to_hub(HF_REPO_ID, token=hf_token)
            tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
            print(f"[OK] Pushed to HuggingFace Hub: {HF_REPO_ID}")
        else:
            print("[WARNING] HF_TOKEN khong set, bo qua push to Hub")
            print("   Dat: export HF_TOKEN='hf_...'")
    except Exception as e:
        print(f"[WARNING] Push to Hub loi: {e}")

    # Merge model (Unsloth ho tro merge 4-bit -> full model)
    try:
        merged_path = OUTPUT_DIR_TRAIN / "merged_model"
        model.save_pretrained_merged(str(merged_path), tokenizer)
        print(f"[OK] Merged model saved: {merged_path}")
    except Exception as e:
        print(f"[WARNING] Merge loi (co the thieu RAM): {e}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 3 – Fine-tuning Qwen2.5-3B (QLoRA + Unsloth)  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 1. Load model
    model, tokenizer = load_model()

    # 2. Prepare data
    dataset = prepare_dataset(tokenizer)
    if dataset is None:
        exit(1)

    # 3. Train
    trainer = train(model, tokenizer, dataset)

    # 4. Save
    save_model(model, tokenizer)

    print("\n[OK] Phase 3 hoan tat! Tiep theo: chay phase4_rag.py")
