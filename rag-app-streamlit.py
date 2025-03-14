import os
import tempfile

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

            supported_formats = ["pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "md", "rtf", "odt", "ods", "odp"]
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

                # تاریخچه گفتگو در یک expander
                with st.expander("گفتگو", expanded=False):
                    chat_container = st.container()
                    with chat_container:
                        for msg in st.session_state.chat_history:
                            if msg["role"] == "user":
                                st.markdown(f'<div class="chat-message user-message">👤 {msg["content"]}</div>',
                                            unsafe_allow_html=True)
                            elif msg["role"] == "assistant":
                                st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"]}</div>',
                                            unsafe_allow_html=True)

                # فرم پرس‌وجو
                with st.form(key="query_form", clear_on_submit=True):
                    query = st.text_area("پرسش خود را وارد کنید:", height=100, key="query_input")
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        top_k = st.slider("تعداد منابع برای بازیابی:", min_value=1, max_value=10, value=4)
                    with col3:
                        submit_button = st.form_submit_button(label="ارسال", type="primary")

                if submit_button and query:
                    st.session_state.chat_history.append({"role": "user", "content": query})

                    progress_placeholder = st.empty()
                    progress_bar = st.progress(0)

                    def update_query_progress(progress, message):
                        progress_bar.progress(progress)
                        progress_placeholder.text(message)

                    with st.spinner("در حال پردازش..."):
                        response = processor.query(query, top_k, update_query_progress)

                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    progress_bar.empty()
                    progress_placeholder.empty()
                    st.rerun()

                # دکمه پاک کردن گفتگو
                if st.button("پاک کردن گفتگو", key="clear_chat"):
                    st.session_state.chat_history = []
                    processor.clearChatHitsory()
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
                for i, file in enumerate(files, 1):
                    file_type = file.get("file_type", "نامشخص")
                    icon = get_file_icon(file_type)

                    metadata = file.get("metadata", {})
                    file_size = metadata.get("file_size_mb", "")
                    size_display = f"{file_size} MB" if file_size else ""

                    data.append({
                        "شماره": i,
                        "نوع": f"{icon} {file_type}",
                        "نام فایل": file["file_name"],
                        "اندازه": size_display,
                        "تعداد صفحات": file["page_count"],
                        "تاریخ پردازش": file["processed_at"]
                    })

                st.table(data)

    finally:
        processor.close()


if __name__ == "__main__":
    main()