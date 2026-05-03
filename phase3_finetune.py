"""
Phase 3 – Fine-tuning Qwen2.5-3B-Instruct với QLoRA trên Colab.

Workflow:
  1. Load base model (4-bit quantized)
  2. Thêm LoRA adapters
  3. Prepare training data (chat format)
  4. Train với SFTTrainer
  5. Save & push lên HuggingFace Hub

Cài đặt (trên Colab):
  !pip install -Uq --no-deps trl peft accelerate bitsandbytes
  !pip install -Uq transformers datasets

Chạy:
  Chạy trên Google Colab với GPU T4
"""

import os
import json
import shutil
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")

# Model config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
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

# Google Drive backup (tự động sao lưu sau mỗi epoch)
DRIVE_BACKUP_DIR = Path("/content/drive/MyDrive/TDTU_Chatbot_Data/finetune")

# System prompt cho chatbot
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về quy chế và sổ tay sinh viên Trường Đại học Tôn Đức Thắng (TDTU).
Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên một cách chính xác, rõ ràng và thân thiện.
Khi trả lời, hãy trích dẫn cụ thể số điều, khoản trong quy chế nếu có.
Nếu không chắc chắn, hãy nói rõ rằng bạn không tìm thấy thông tin."""


# ══════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ══════════════════════════════════════════════════════════

def load_model():
    """Load Qwen2.5-3B-Instruct với 4-bit quantization + LoRA (Thuần Hugging Face)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    
    print("[INIT] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[INIT] Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if LOAD_IN_4BIT else None,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Chuẩn bị cho 4-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("[INIT] Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # In thông tin model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model loaded!")
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model, tokenizer


# ══════════════════════════════════════════════════════════
# 2. PREPARE DATASET
# ══════════════════════════════════════════════════════════

def prepare_dataset(tokenizer):
    """Chuyển QA pairs sang chat format cho SFTTrainer"""
    
    qa_train_path = BASE_DIR / "processed" / "qa_train.json"
    if not qa_train_path.exists():
        print(f"[ERROR] Khong tim thay {qa_train_path}")
        print("   Chay phase2_process.py truoc!")
        return None
    
    with open(qa_train_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    
    print(f"[INFO] Loaded {len(qa_pairs)} QA pairs")
    
    # Chuyển sang chat format
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
        
        # Truncate nếu quá dài (thay cho max_seq_length của SFTConfig)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > MAX_SEQ_LENGTH:
            tokens = tokens[:MAX_SEQ_LENGTH]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
        
        formatted_data.append({"text": text})
    
    # Tạo HuggingFace Dataset
    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data)
    
    print(f"[OK] Dataset: {len(dataset)} samples")
    print(f"   Ví dụ: {dataset[0]['text'][:200]}...")
    
    return dataset


# ══════════════════════════════════════════════════════════
# 3. DRIVE BACKUP CALLBACK
# ══════════════════════════════════════════════════════════

class DriveBackupCallback:
    """Tự động copy checkpoint lên Google Drive sau mỗi epoch"""
    
    def on_save(self, args, state, control, **kwargs):
        """Được gọi sau mỗi lần save checkpoint"""
        if not DRIVE_BACKUP_DIR.parent.exists():
            print("[WARNING] Google Drive chua mount, bo qua backup")
            return
        
        try:
            DRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            
            # Tìm checkpoint mới nhất
            checkpoint_dirs = sorted(
                Path(args.output_dir).glob("checkpoint-*"),
                key=lambda x: x.stat().st_mtime
            )
            if checkpoint_dirs:
                latest = checkpoint_dirs[-1]
                dest = DRIVE_BACKUP_DIR / latest.name
                if dest.exists():
                    shutil.rmtree(str(dest))
                shutil.copytree(str(latest), str(dest))
                print(f"[BACKUP] Saved to Drive: {dest}")
        except Exception as e:
            print(f"[WARNING] Drive backup loi: {e}")


# ══════════════════════════════════════════════════════════
# 4. TRAIN
# ══════════════════════════════════════════════════════════

def train(model, tokenizer, dataset):
    """Fine-tune model bằng SFTTrainer (tương thích mọi phiên bản trl)"""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    OUTPUT_DIR_TRAIN.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect precision: bf16 không dùng GradScaler nên tránh crash
    # fp16 GradScaler KHÔNG tương thích với bitsandbytes 4-bit (luôn sinh bf16 gradients)
    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"   [INFO] Mixed precision: {'bf16' if use_bf16 else 'fp32 (no mixed precision)'}")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,           # KHÔNG dùng fp16 với bitsandbytes 4-bit
        bf16=use_bf16,        # bf16 nếu GPU hỗ trợ (không dùng GradScaler)
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=str(OUTPUT_DIR_TRAIN),
        save_strategy="epoch",
        report_to="none",
    )
    
    # Auto-detect API + thêm callback backup Drive
    from transformers import TrainerCallback
    
    class _DriveBackup(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            DriveBackupCallback().on_save(args, state, control, **kwargs)
    
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[_DriveBackup()],
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[_DriveBackup()],
        )
    
    print("\n[TRAIN] Bat dau fine-tuning...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size (effective): {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    # GPU memory trước training
    import torch
    if torch.cuda.is_available():
        print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB used")
    
    # Resume từ checkpoint nếu có
    resume_dir = None
    checkpoints = sorted(OUTPUT_DIR_TRAIN.glob("checkpoint-*")) if OUTPUT_DIR_TRAIN.exists() else []
    if not checkpoints:
        # Thử tìm trên Drive
        drive_checkpoints = sorted(DRIVE_BACKUP_DIR.glob("checkpoint-*")) if DRIVE_BACKUP_DIR.exists() else []
        if drive_checkpoints:
            latest_drive = drive_checkpoints[-1]
            local_copy = OUTPUT_DIR_TRAIN / latest_drive.name
            if not local_copy.exists():
                shutil.copytree(str(latest_drive), str(local_copy))
                print(f"[RESUME] Restored checkpoint from Drive: {latest_drive.name}")
            resume_dir = str(local_copy)
    else:
        resume_dir = str(checkpoints[-1])
        print(f"[RESUME] Found local checkpoint: {checkpoints[-1].name}")
    
    # Train (resume nếu có checkpoint)
    stats = trainer.train(resume_from_checkpoint=resume_dir)
    
    print(f"\n[OK] Training hoan tat!")
    print(f"   Total steps: {stats.global_step}")
    print(f"   Training loss: {stats.training_loss:.4f}")
    
    return trainer


# ══════════════════════════════════════════════════════════
# 4. SAVE & PUSH
# ══════════════════════════════════════════════════════════

def save_model(model, tokenizer):
    """Lưu LoRA adapter + push lên HuggingFace Hub"""
    
    # Lưu local
    lora_path = OUTPUT_DIR_TRAIN / "lora_adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"[OK] LoRA adapter saved: {lora_path}")
    
    # Push lên HuggingFace Hub (optional)
    try:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            model.push_to_hub(HF_REPO_ID, token=hf_token)
            tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
            print(f"[OK] Pushed to HuggingFace Hub: {HF_REPO_ID}")
        else:
            print("[WARNING] HF_TOKEN khong set, bo qua push to Hub")
            print("   Đặt: export HF_TOKEN='hf_...'")
    except Exception as e:
        print(f"[WARNING] Push to Hub loi: {e}")
    
    # Không hỗ trợ merge model tự động bằng PEFT khi đang load 4-bit,
    # Phase 4 (RAG) sẽ tự động load base model + adapter riêng nên không cần thiết.
    
    # Backup LoRA adapter lên Drive
    try:
        if DRIVE_BACKUP_DIR.parent.exists():
            drive_lora = DRIVE_BACKUP_DIR / "lora_adapter"
            if drive_lora.exists():
                shutil.rmtree(str(drive_lora))
            shutil.copytree(str(lora_path), str(drive_lora))
            print(f"[BACKUP] LoRA adapter saved to Drive: {drive_lora}")
    except Exception as e:
        print(f"[WARNING] Drive backup loi: {e}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 3 – Fine-tuning Qwen2.5-3B (QLoRA)            ║")
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
