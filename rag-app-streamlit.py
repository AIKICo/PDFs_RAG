import os
import tempfile

import pandas as pd
import streamlit as st
import torch

from DocumentProcess.DocumentProcess import DocumentProcess

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

torch.cuda.is_available = lambda: False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(
    page_title="سیستم RAG اسناد",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استفاده از فونت Vazirmatn از لینک ایمن و اعمال آن به کل رابط کاربری
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/npm/vazirmatn@33.0.3/Vazirmatn-font-face.css');

    html, body {
        direction: rtl;
        font-family: 'Vazirmatn', sans-serif !important;
    }
    .stApp {
        direction: rtl;
        font-family: 'Vazirmatn', sans-serif !important;
    }
    * {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #fafafa;
    }
    .chat-message {
        direction: rtl;
        text-align: right;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        max-width: 80%;
    }
    .user-message {
        background-color: #d4eaff;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #e6e6e6;
        margin-right: auto;
    }
    /* سایر استایل‌ها */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Vazirmatn', sans-serif !important;
        font-weight: bold;
    }
    div[data-testid="stExpander"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stSelectSlider"] label {
        text-align: right;
        width: 100%;
    }
    
    div[data-testid="stNumberInput"] div[data-testid="stWidgetLabel"] {
        direction: rtl;
        text-align: right;
    }
    
    /* استایل برای اعداد در RTL */
    div[data-testid="stNumberInput"] div[aria-label="range"] {
        direction: ltr;
        text-align: left;
    }
    div[data-testid="stExpander"] {
    border: 1px solid #eaeaea;
    border-radius: 8px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

    div[data-testid="stExpander"] > div[role="button"] {
        font-size: 0.95em;
        font-weight: 500;
        color: #333;
        padding: 8px 12px;
    }
    
    /* تنظیم جهت و حاشیه‌های محتوای داخل expander */
    div[data-testid="stExpander"] > div[data-testid="stExpanderContent"] {
        direction: rtl;
        text-align: right;
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)


def get_file_icon(file_type):
    icons = {
        'pdf': '📄', 'docx': '📝', 'doc': '📝', 'xlsx': '📊', 'xls': '📊',
        'pptx': '📊', 'ppt': '📊', 'txt': '📋', 'md': '📝', 'rtf': '📄',
        'odt': '📝', 'ods': '📊', 'odp': '📊',
    }
    return icons.get(file_type.lower(), '📁')


def save_uploaded_file(uploaded_file):
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"خطا در ذخیره فایل: {e}")
        return None


def main():
    st.sidebar.title("سیستم RAG اسناد")
    st.sidebar.subheader("با پشتیبانی از زبان فارسی")

    page = st.sidebar.radio("صفحات", ["پردازش فایل‌ها", "پرسش و پاسخ", "فایل‌های پردازش شده"])

    st.sidebar.subheader("تنظیمات مدل")
    model_name = st.sidebar.text_input("مدل LLM (gemma3)", value="gemma3")
    embeddings_model = st.sidebar.text_input("مدل Embeddings",
                                             value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    processor = DocumentProcess(model_name=model_name, embeddings_model=embeddings_model)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    try:
        if page == "پردازش فایل‌ها":
            st.title("پردازش اسناد")
            st.write("فایل‌های خود را آپلود کنید تا پردازش شوند.")

            supported_formats = ["pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "md", "rtf", "odt", "ods",
                                 "odp"]
            formats_display = ", ".join([f".{fmt}" for fmt in supported_formats])

            uploaded_files = st.file_uploader(f"فایل‌ها را انتخاب کنید ({formats_display})",
                                              type=supported_formats,
                                              accept_multiple_files=True)

            col1, col2 = st.columns([4, 1])
            with col2:
                process_button = st.button("پردازش فایل‌ها", type="primary")

            if uploaded_files and process_button:
                file_paths = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        file_paths.append(file_path)

                if file_paths:
                    def update_progress(current, total, message=""):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)

                    result = processor.process_documents(file_paths, update_progress)

                    progress_bar.progress(1.0)
                    status_text.text("پردازش کامل شد!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("فایل‌های پردازش شده", result["processed"])
                    with col2:
                        st.metric("فایل‌های رد شده", result["skipped"])
                    with col3:
                        st.metric("تکه‌های متنی جدید", result["new_chunks"])

                    if result["errors"]:
                        st.error("خطاها:")
                        for error in result["errors"]:
                            st.write(f"- {error}")

                    st.success(f"{result['processed']} فایل با موفقیت پردازش شد و به پایگاه داده اضافه گردید.")

        elif page == "پرسش و پاسخ":
            st.title("پرسش و پاسخ از اسناد")

            files = processor.list_processed_files()
            if not files:
                st.warning("هیچ فایلی در پایگاه داده موجود نیست. لطفاً ابتدا فایل‌ها را پردازش کنید.")
            else:
                st.write(f"{len(files)} فایل در پایگاه داده موجود است.")

                # نمایش تاریخچه گفتگو بدون استفاده از expander
                st.subheader("تاریخچه گفتگو")
                chat_container = st.container()
                with chat_container:
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            st.markdown(f'<div class="chat-message user-message">👤 {msg["content"]}</div>',
                                        unsafe_allow_html=True)
                        elif msg["role"] == "assistant":
                            # بررسی اینکه پاسخ به صورت دیکشنری است یا متن ساده
                            if isinstance(msg["content"], dict):
                                answer = msg["content"].get("answer", "")
                                sources = msg["content"].get("sources", [])

                                # نمایش پاسخ اصلی
                                st.markdown(f'<div class="chat-message assistant-message">🤖 {answer}</div>',
                                            unsafe_allow_html=True)

                                # نمایش منابع به صورت کلپسیبل بدون استفاده از expander
                                if sources:
                                    # ایجاد یک مشخصه یکتا برای هر پیام
                                    message_id = f"msg_{st.session_state.chat_history.index(msg)}"

                                    # استفاده از دکمه برای نمایش/پنهان کردن منابع
                                    if f"{message_id}_show_sources" not in st.session_state:
                                        st.session_state[f"{message_id}_show_sources"] = False

                                    if st.button(f"📚 نمایش منابع", key=f"btn_{message_id}"):
                                        st.session_state[f"{message_id}_show_sources"] = not st.session_state[
                                            f"{message_id}_show_sources"]

                                    # نمایش منابع اگر دکمه فعال شده باشد
                                    if st.session_state[f"{message_id}_show_sources"]:
                                        with st.expander("📚 منابع استفاده شده"):
                                            st.markdown("""
                                               <style>
                                                   .sources-container {
                                                       font-size: 0.9em;
                                                       direction: rtl;
                                                       text-align: right;
                                                   }
                                                   .source-item {
                                                       border-bottom: 1px solid #eee;
                                                       padding-bottom: 10px;
                                                       margin-bottom: 10px;
                                                   }
                                                   .source-item:last-child {
                                                       border-bottom: none;
                                                   }
                                                   .source-title {
                                                       font-weight: bold;
                                                       font-size: 1em;
                                                       margin-bottom: 5px;
                                                   }
                                                   .source-score, .source-path {
                                                       font-size: 0.85em;
                                                       color: #666;
                                                       margin-bottom: 3px;
                                                   }
                                               </style>
                                               <div class="sources-container">
                                               """, unsafe_allow_html=True)
                                            for i, source in enumerate(sources, 1):
                                                title = source.get('title', 'بدون عنوان')
                                                score = source.get('score', 0.0)
                                                path = source.get('source', 'نامشخص')
                                                st.markdown(f"""
                                                            <div class="source-item">
                                                                <div class="source-title">منبع {i}: {title}</div>
                                                                <div class="source-score">امتیاز ارتباط: {score:.2f}</div>
                                                                <div class="source-path">مسیر: {path}</div>
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                # نمایش پاسخ به صورت متن ساده (برای سازگاری با پاسخ‌های قدیمی)
                                st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"]}</div>',
                                            unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # فرم پرس‌وجو
                with st.form(key="query_form", clear_on_submit=True):
                    query = st.text_area("پرسش خود را وارد کنید:", height=100, key="query_input")
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        top_k = st.number_input("تعداد منابع برای بازیابی:",
                                                min_value=1, max_value=10, value=4, step=1)
                    with col3:
                        submit_button = st.form_submit_button(label="ارسال", type="primary")

                # پردازش پرسش
                if submit_button and query:
                    # افزودن پرسش به تاریخچه
                    st.session_state.chat_history.append({"role": "user", "content": query})

                    # ایجاد نشانگر پیشرفت
                    progress_container = st.container()
                    with progress_container:
                        progress_text = st.empty()
                        progress_bar = st.progress(0)

                        def update_query_progress(progress, message):
                            progress_bar.progress(progress)
                            progress_text.text(message)

                        # پردازش پرسش
                        with st.spinner("در حال پردازش..."):
                            response = processor.query(query, top_k, update_query_progress)

                        # افزودن پاسخ به تاریخچه
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

                        # پاک کردن نشانگر پیشرفت
                        progress_bar.empty()
                        progress_text.empty()

                        # به‌روزرسانی صفحه
                        st.rerun()

                # دکمه پاک کردن گفتگو
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button("پاک کردن گفتگو", key="clear_chat"):
                        st.session_state.chat_history = []
                        processor.clearChatHitsory()  # اصلاح نام متد
                        st.success("گفتگو پاک شد.")
                        st.rerun()

        elif page == "فایل‌های پردازش شده":
            st.title("فایل‌های پردازش شده")

            if 'refresh_files' not in st.session_state:
                st.session_state.refresh_files = False

            files = processor.list_processed_files()
            if not files:
                st.info("هیچ فایلی پردازش نشده است.")
            else:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{len(files)} فایل در پایگاه داده موجود است.")
                with col2:
                    if st.button("🔄 تازه‌سازی"):
                        st.session_state.refresh_files = True
                        st.rerun()

                data = []
                # اضافه کردن ستونی برای عملیات حذف
                for i, file in enumerate(files, 1):
                    file_type = file.get("file_type", "نامشخص")
                    icon = get_file_icon(file_type)

                    metadata = file.get("metadata", {})
                    file_size = metadata.get("file_size_mb", "")
                    size_display = f"{file_size} MB" if file_size else ""

                    # محاسبه file_hash برای حذف
                    file_path = file.get("file_path", "")
                    file_hash = processor._calculate_file_hash(file_path) if file_path and os.path.exists(
                        file_path) else ""

                    data.append({
                        "شماره": i,
                        "نوع": f"{icon} {file_type}",
                        "نام فایل": file["file_name"],
                        "اندازه": size_display,
                        "تعداد صفحات": file["page_count"],
                        "تاریخ پردازش": file["processed_at"],
                        "عملیات": file_hash  # ذخیره هش فایل برای استفاده در حذف
                    })

                # نمایش جدول با دکمه‌های حذف
                df = pd.DataFrame(data)
                for i, row in df.iterrows():
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 2, 1, 1, 2, 1])
                    with col1:
                        st.write(row["شماره"])
                    with col2:
                        st.write(row["نوع"])
                    with col3:
                        st.write(row["نام فایل"])
                    with col4:
                        st.write(row["اندازه"])
                    with col5:
                        st.write(row["تعداد صفحات"])
                    with col6:
                        st.write(row["تاریخ پردازش"])
                    with col7:
                        if st.button("🗑️ حذف", key=f"delete_{row['عملیات']}"):
                            if processor.remove_document(row["عملیات"]):
                                st.success(f"فایل «{row['نام فایل']}» با موفقیت حذف شد.")
                                st.session_state.refresh_files = True
                                st.rerun()
                            else:
                                st.error(f"خطا در حذف فایل «{row['نام فایل']}»")

                # st.table(data)

    finally:
        processor.close()


if __name__ == "__main__":
    main()
