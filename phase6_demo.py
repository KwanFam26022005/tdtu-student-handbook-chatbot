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

import gradio as gr
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")


# ══════════════════════════════════════════════════════════
# TDTU BRAND: Official Color System (extracted from logo)
# ══════════════════════════════════════════════════════════
TDTU_BLUE       = "#1565C0"   # TDTU BRAND: primary — dominant (60%+)
TDTU_RED        = "#E8293A"   # TDTU BRAND: accent — sparingly (badges, highlights)
TDTU_BLUE_DARK  = "#0D47A1"   # TDTU BRAND: hover state for blue
TDTU_BLUE_LIGHT = "#E8F0FB"   # TDTU BRAND: surfaces, input backgrounds
TDTU_WHITE      = "#FFFFFF"   # TDTU BRAND: base background
TDTU_TEXT       = "#0A2A5E"   # TDTU BRAND: body text (navy)
TDTU_TEXT_MUTED = "#5A7AB0"   # TDTU BRAND: placeholder, secondary labels

# TDTU BRAND: Vietnamese UI text — TDTU official tone
UI_TITLE        = "🤖 Trợ Lý AI — Sổ Tay Sinh Viên TDTU"
UI_GREETING     = "Xin chào! Tôi là trợ lý AI của Trường Đại học Tôn Đức Thắng."
UI_SUBGREETING  = "Bạn có thể hỏi tôi về quy chế, học phí, thời khóa biểu và nội quy nhà trường."
UI_PLACEHOLDER  = "Nhập câu hỏi của bạn..."
UI_BUTTON       = "GỬI  ➤"
UI_LOADING      = "⏳ Đang tra cứu sổ tay sinh viên..."
UI_NO_ANSWER    = "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Vui lòng liên hệ phòng ban liên quan."
UI_FOOTER       = "© Trường Đại học Tôn Đức Thắng (TDTU)"


# ══════════════════════════════════════════════════════════
# TDTU BRAND: Complete CSS stylesheet
# ══════════════════════════════════════════════════════════
tdtu_css = f"""
/* ========================================================
   TDTU BRAND: CSS Variables — Official Color System
   ======================================================== */
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;700;800&display=swap');

:root {{
    --tdtu-blue:        {TDTU_BLUE};
    --tdtu-red:         {TDTU_RED};
    --tdtu-blue-dark:   {TDTU_BLUE_DARK};
    --tdtu-blue-light:  {TDTU_BLUE_LIGHT};
    --tdtu-white:       {TDTU_WHITE};
    --tdtu-text:        {TDTU_TEXT};
    --tdtu-text-muted:  {TDTU_TEXT_MUTED};
}}

/* TDTU BRAND: Global font — Be Vietnam Pro */
* {{
    font-family: 'Be Vietnam Pro', 'Roboto', sans-serif !important;
}}

/* TDTU BRAND: Container max-width */
.gradio-container {{
    max-width: 960px !important;
    margin: 0 auto !important;
    background: var(--tdtu-white) !important;
}}

/* TDTU BRAND: Header block — TDTU BLUE dominant */
.tdtu-header {{
    background: linear-gradient(135deg, {TDTU_BLUE} 0%, {TDTU_BLUE_DARK} 100%);
    color: {TDTU_WHITE};
    padding: 28px 24px 22px;
    border-radius: 16px;
    margin-bottom: 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(21, 101, 192, 0.3);
}}

.tdtu-header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}}

.tdtu-header h1 {{
    margin: 0 0 4px 0;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: {TDTU_WHITE} !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}

/* TDTU BRAND: Subheader — uppercase like logo */
.tdtu-header .tdtu-uni-name {{
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    opacity: 0.92;
    margin-bottom: 10px;
}}

.tdtu-header .tdtu-greeting {{
    font-size: 0.95rem;
    font-weight: 400;
    opacity: 0.9;
    margin-top: 8px;
    line-height: 1.5;
}}

/* TDTU BRAND: Accent divider — RED accent line (thin) */
.tdtu-divider {{
    width: 60px;
    height: 3px;
    background: {TDTU_RED};
    margin: 12px auto;
    border-radius: 2px;
}}

/* TDTU BRAND: Controls row */
.tdtu-controls {{
    background: {TDTU_BLUE_LIGHT};
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(21, 101, 192, 0.12);
}}

.tdtu-controls label {{
    color: {TDTU_TEXT} !important;
    font-weight: 500 !important;
}}

/* TDTU BRAND: Chat bubbles — bot (light surface + blue left border) */
.message.bot .message-bubble-border {{
    border: none !important;
}}

.message.bot {{
    background: {TDTU_BLUE_LIGHT} !important;
    border-left: 3px solid {TDTU_BLUE} !important;
    border-radius: 4px 16px 16px 16px !important;
    color: {TDTU_TEXT} !important;
    padding: 14px 16px !important;
}}

/* TDTU BRAND: Chat bubbles — user (TDTU BLUE bg, white text) */
.message.user {{
    background: {TDTU_BLUE} !important;
    color: {TDTU_WHITE} !important;
    border-radius: 16px 4px 16px 16px !important;
    padding: 14px 16px !important;
    box-shadow: 0 2px 8px rgba(21, 101, 192, 0.25);
}}

/* Gradio chatbot message styling overrides */
.chatbot .message-wrap .message {{
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}}

/* TDTU BRAND: Input textbox */
.gradio-container textarea,
.gradio-container input[type="text"] {{
    background: {TDTU_BLUE_LIGHT} !important;
    border: 2px solid transparent !important;
    border-radius: 12px !important;
    color: {TDTU_TEXT} !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {{
    border-color: {TDTU_BLUE} !important;
    box-shadow: 0 0 0 3px rgba(21, 101, 192, 0.15) !important;
    outline: none !important;
}}

.gradio-container textarea::placeholder,
.gradio-container input[type="text"]::placeholder {{
    color: {TDTU_TEXT_MUTED} !important;
}}

/* TDTU BRAND: Primary button — TDTU BLUE */
.primary.svelte-cmf5ev,
button.primary {{
    background: {TDTU_BLUE} !important;
    color: {TDTU_WHITE} !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px;
    padding: 10px 24px !important;
    box-shadow: 0 2px 10px rgba(21, 101, 192, 0.25) !important;
    transition: all 0.2s ease !important;
}}

button.primary:hover {{
    background: {TDTU_BLUE_DARK} !important;
    box-shadow: 0 4px 16px rgba(13, 71, 161, 0.35) !important;
    transform: translateY(-1px);
}}

/* TDTU BRAND: Secondary buttons */
button.secondary {{
    background: {TDTU_WHITE} !important;
    color: {TDTU_BLUE} !important;
    border: 1.5px solid {TDTU_BLUE} !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}}

button.secondary:hover {{
    background: {TDTU_BLUE_LIGHT} !important;
}}

/* TDTU BRAND: Example prompts */
.examples-row button {{
    background: {TDTU_WHITE} !important;
    border: 1.5px solid {TDTU_BLUE} !important;
    color: {TDTU_TEXT} !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}}

.examples-row button:hover {{
    background: {TDTU_BLUE_LIGHT} !important;
    border-color: {TDTU_BLUE_DARK} !important;
}}

/* TDTU BRAND: Badge accent — RED (only here) */
.tdtu-badge {{
    display: inline-block;
    background: {TDTU_RED};
    color: {TDTU_WHITE};
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* TDTU BRAND: Footer */
.tdtu-footer {{
    text-align: center;
    padding: 16px 12px;
    color: {TDTU_TEXT_MUTED};
    font-size: 0.8rem;
    border-top: 2px solid {TDTU_RED};
    margin-top: 16px;
}}

.tdtu-footer strong {{
    color: {TDTU_BLUE};
}}

/* TDTU BRAND: Scrollbar styling */
::-webkit-scrollbar {{
    width: 6px;
}}
::-webkit-scrollbar-track {{
    background: {TDTU_BLUE_LIGHT};
}}
::-webkit-scrollbar-thumb {{
    background: {TDTU_BLUE};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {TDTU_BLUE_DARK};
}}

/* TDTU BRAND: Loading spinner override */
.wrap.svelte-cmf5ev {{
    border-color: {TDTU_BLUE} !important;
}}

/* TDTU BRAND: Checkbox accent */
input[type="checkbox"]:checked {{
    background-color: {TDTU_BLUE} !important;
    border-color: {TDTU_BLUE} !important;
}}

/* TDTU BRAND: Smooth global transitions */
* {{
    transition-property: background-color, border-color, color, box-shadow;
    transition-duration: 0.15s;
    transition-timing-function: ease;
}}
"""


def create_demo():
    """Tạo Gradio demo app với TDTU brand identity"""

    # Lazy load pipeline (tránh load model khi chưa cần)
    pipeline_cache = {}

    def get_pipeline(use_finetuned):
        key = f"ft_{use_finetuned}"
        if key not in pipeline_cache:
            # Giải phóng pipeline cũ để tránh OOM trên Colab T4 (15GB VRAM)
            for old_key in list(pipeline_cache.keys()):
                print(f"  [MEMORY] Đang dọn dẹp bộ nhớ pipeline cũ: {old_key}")
                pipeline_cache[old_key].cleanup()
                del pipeline_cache[old_key]
                
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

    # ══════════════════════════════════════════════════════════
    # TDTU BRAND: Build Gradio UI
    # ══════════════════════════════════════════════════════════

    with gr.Blocks(
        title=UI_TITLE,  # TDTU BRAND: browser tab title
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            font=("Be Vietnam Pro", "Roboto", "sans-serif"),
            font_mono=("Be Vietnam Pro", "monospace"),
        ),
        css=tdtu_css,
    ) as demo:

        # TDTU BRAND: Header — TDTU BLUE gradient, uppercase uni name, greeting
        gr.HTML(f"""
        <div class="tdtu-header">
            <div class="tdtu-uni-name">ĐẠI HỌC TÔN ĐỨC THẮNG</div>
            <h1>{UI_TITLE}</h1>
            <div class="tdtu-divider"></div>
            <div class="tdtu-greeting">
                {UI_GREETING}<br/>
                <span style="opacity:0.8;">{UI_SUBGREETING}</span>
            </div>
            <div style="margin-top:10px;">
                <span class="tdtu-badge">AI CHATBOT</span>
            </div>
        </div>
        """)

        # TDTU BRAND: Controls row — toggle RAG / fine-tuned model
        with gr.Row(elem_classes="tdtu-controls"):
            use_rag = gr.Checkbox(
                label="🔍 Sử dụng RAG (tìm kiếm trong quy chế)",
                value=True
            )
            use_finetuned = gr.Checkbox(
                label="🎯 Dùng model Fine-tuned",
                value=True
            )

        # TDTU BRAND: Chat interface with Vietnamese labels
        # Note: retry_btn, undo_btn, clear_btn removed in Gradio 6.x
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
            textbox=gr.Textbox(
                placeholder=UI_PLACEHOLDER,  # TDTU BRAND: Vietnamese placeholder
                container=False,
                scale=7,
            ),
            submit_btn=UI_BUTTON,      # TDTU BRAND: Vietnamese submit button
        )

        # TDTU BRAND: Footer — RED accent divider + muted text
        gr.HTML(f"""
        <div class="tdtu-footer">
            <p>⚠️ Chatbot chỉ trả lời dựa trên quy chế đã có trong hệ thống.
            Vui lòng kiểm tra lại thông tin quan trọng với phòng Công tác Sinh viên.</p>
            <p style="margin-top: 8px;">
                <strong>{UI_FOOTER}</strong><br/>
                Powered by RAG + Fine-tuned Qwen2.5-3B
            </p>
        </div>
        """)

    return demo


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # TDTU BRAND: Console banner
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🤖 TRỢ LÝ AI — SỔ TAY SINH VIÊN TDTU               ║")
    print("║   PHASE 6 – Gradio Demo (TDTU Brand)                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    demo = create_demo()
    demo.launch(
        share=True,      # Tạo public link (cho Colab)
        server_name="0.0.0.0",
        server_port=7860,
    )
