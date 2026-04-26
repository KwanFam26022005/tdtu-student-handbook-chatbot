"""
Phase 6 – Gradio Demo: Giao diện chatbot sổ tay sinh viên TDTU.

Features:
  - Chat interface đẹp
  - Toggle RAG on/off
  - Toggle base vs fine-tuned model
  - Hiển thị nguồn trích dẫn
  - Hiển thị retrieval scores

Cài đặt:
  pip install gradio

Chạy:
  python phase6_demo.py
  
Trên Colab:
  %run phase6_demo.py
  # Gradio sẽ tạo public link tự động
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")


def create_demo():
    """Tạo Gradio demo app"""
    import gradio as gr
    
    # Lazy load pipeline (tránh load model khi chưa cần)
    pipeline_cache = {}
    
    def get_pipeline(use_finetuned):
        key = f"ft_{use_finetuned}"
        if key not in pipeline_cache:
            from phase4_rag import RAGPipeline
            pipeline_cache[key] = RAGPipeline(use_finetuned=use_finetuned)
        return pipeline_cache[key]
    
    def chat(message, history, use_rag, use_finetuned):
        """Handler cho chat interface"""
        pipeline = get_pipeline(use_finetuned)
        result = pipeline.answer(message, use_rag=use_rag)
        
        answer = result["answer"]
        
        # Thêm nguồn trích dẫn
        if result["sources"]:
            sources = list(set(result["sources"][:3]))
            answer += f"\n\n📚 **Nguồn:** {', '.join(sources)}"
        
        if result["sections"]:
            sections = [s for s in result["sections"][:3] if s]
            if sections:
                answer += f"\n📋 **Phần:** {', '.join(sections)}"
        
        return answer
    
    # ── Build UI ──
    with gr.Blocks(
        title="🎓 Chatbot Sổ Tay Sinh Viên TDTU",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 900px !important; }
        .header { text-align: center; padding: 20px; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🎓 Chatbot Sổ Tay Sinh Viên TDTU</h1>
            <p>Hỏi đáp về quy chế, quy định sinh viên Trường ĐH Tôn Đức Thắng</p>
            <p><em>Powered by RAG + Fine-tuned Qwen2.5-3B</em></p>
        </div>
        """)
        
        with gr.Row():
            use_rag = gr.Checkbox(
                label="🔍 Sử dụng RAG (tìm kiếm trong quy chế)", 
                value=True
            )
            use_finetuned = gr.Checkbox(
                label="🎯 Dùng model Fine-tuned", 
                value=True
            )
        
        chatbot = gr.ChatInterface(
            fn=chat,
            additional_inputs=[use_rag, use_finetuned],
            examples=[
                ["Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp?"],
                ["Điều kiện để được xét học bổng là gì?"],
                ["Nếu bị cảnh báo học vụ thì sinh viên phải làm gì?"],
                ["Thủ tục xin bảo lưu kết quả học tập như thế nào?"],
                ["Sinh viên bị xử lý kỷ luật trong trường hợp nào?"],
                ["Quy định về tiếng Anh đầu ra như thế nào?"],
                ["Điều kiện miễn học phí cho sinh viên?"],
                ["Điểm rèn luyện được tính như thế nào?"],
            ],
            title="",
            retry_btn="🔄 Thử lại",
            undo_btn="↩️ Hoàn tác",
            clear_btn="🗑️ Xóa chat",
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 10px; color: #666;">
            <p>⚠️ Chatbot chỉ trả lời dựa trên quy chế đã có trong hệ thống. 
            Vui lòng kiểm tra lại thông tin quan trọng với phòng Công tác Sinh viên.</p>
        </div>
        """)
    
    return demo


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 6 – Gradio Demo                                ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    demo = create_demo()
    demo.launch(
        share=True,      # Tạo public link (cho Colab)
        server_name="0.0.0.0",
        server_port=7860,
    )
