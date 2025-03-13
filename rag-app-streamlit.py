import os
import tempfile

import streamlit as st
import torch
from langchain.chains import RetrievalQA

from FileProcessing.DocumentProcess import DocumentProcess

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

st.markdown("""
<style>
    /* تنظیمات عمومی RTL */
    .stApp {
        direction: rtl;
    }

    /* اصلاح نوار اسکرول */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* اصلاح جزئیات رابط کاربری */
    .element-container div {
        direction: rtl;
    }

    /* اصلاح دکمه‌ها */
    .stButton button {
        direction: rtl;
        text-align: center;
        width: 100%;
    }

    /* اصلاح جداول */
    th, td {
        text-align: right !important;
    }

    /* اصلاح فرم‌ها */
    div[data-testid="stForm"] {
        direction: rtl;
        text-align: right;
    }

    /* اصلاح متن‌ها */
    div[data-testid="stMarkdown"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stText"] {
        direction: rtl;
        text-align: right;
    }

    /* اصلاح جدول‌ها */
    div[data-testid="stTable"] th, div[data-testid="stTable"] td {
        text-align: right !important;
    }

    /* اصلاح آپلودر فایل */
    div[data-testid="stFileUploader"] {
        direction: rtl;
    }
    div[data-testid="stFileUploader"] > label {
        direction: rtl;
        text-align: right;
    }

    /* اصلاح فیلدهای متنی */
    div[data-testid="stTextArea"] {
        direction: rtl;
    }
    div[data-testid="stTextArea"] label {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stTextInput"] {
        direction: rtl;
    }
    div[data-testid="stTextInput"] label {
        direction: rtl;
        text-align: right;
    }

    /* اصلاح نوار کناری */
    div[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    section[data-testid="stSidebar"] > div {
        direction: rtl;
    }

    /* اصلاح expander */
    div[data-testid="stExpander"] {
        direction: rtl;
    }

    /* اصلاح اسلایدر */
    div[data-testid="stSlider"] {
        direction: rtl;
    }
    div[data-testid="stSlider"] > div {
        direction: rtl;
    }

    /* اصلاح فونت */
    * {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
    }

    /* اصلاح متریک‌ها */
    div[data-testid="stMetric"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stMetric"] label {
        text-align: right;
    }

    /* اصلاح پیام‌های خطا و هشدار */
    div[data-testid="stAlert"] {
        direction: rtl;
        text-align: right;
    }

    /* اصلاح دکمه‌های رادیویی */
    div[data-testid="stRadio"] {
        direction: rtl;
    }
    div[data-testid="stRadio"] label {
        text-align: right;
    }

    /* اصلاح باکس‌های انتخاب */
    div[data-testid="stCheckbox"] {
        direction: rtl;
    }
    div[data-testid="stCheckbox"] label {
        text-align: right;
    }

    /* اصلاح پیشرفت‌بار */
    div[data-testid="stProgressBar"] {
        direction: ltr;
    }

    /* اصلاح دکمه‌های منو */
    button[kind="primary"], button[kind="secondary"] {
        width: 100%;
        text-align: center;
    }

    /* اصلاح پاسخ‌ها */
    div.response-box {
        direction: rtl;
        text-align: right;
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
        line-height: 1.6;
    }

    /* اصلاح منابع */
    div.source-item {
        direction: rtl;
        text-align: right;
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }

    /* اصلاح فونت عنوان‌ها */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
        font-weight: bold;
    }

    /* اصلاح دکمه‌های نوار کناری */
    .css-1oe5cao {
        padding-right: 1rem !important;
        padding-left: 1rem !important;
    }
</style>

<!-- افزودن فونت وزیرمتن -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css">
""", unsafe_allow_html=True)


def get_file_icon(file_type):
    icons = {
        'pdf': '📄',
        'docx': '📝',
        'doc': '📝',
        'xlsx': '📊',
        'xls': '📊',
        'pptx': '📊',
        'ppt': '📊',
        'txt': '📋',
        'md': '📝',
        'rtf': '📄',
        'odt': '📝',
        'ods': '📊',
        'odp': '📊',
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
                st.warning("هیچ فایلی در پایگاه داده موجود نیست. لطفا ابتدا فایل‌ها را پردازش کنید.")
            else:
                st.write(f"{len(files)} فایل در پایگاه داده موجود است.")

                query = st.text_area("پرسش خود را وارد کنید:", height=100)

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    top_k = st.slider("تعداد منابع برای بازیابی:", min_value=1, max_value=10, value=4)
                with col3:
                    submit_button = st.button("ارسال پرسش", type="primary")

                if submit_button:
                    if not query:
                        st.warning("لطفا یک پرسش وارد کنید.")
                    else:
                        progress_placeholder = st.empty()
                        progress_bar = st.progress(0)

                        def update_query_progress(progress, message):
                            progress_bar.progress(progress)
                            progress_placeholder.text(message)

                        with st.spinner("در حال پردازش پرسش..."):
                            response = processor.query(query, top_k, update_query_progress)

                        st.subheader("پاسخ:")

                        parts = response.split("\n\nمنابع:\n")
                        answer = parts[0].replace("پاسخ: ", "")
                        references = parts[1] if len(parts) > 1 else ""

                        st.markdown(
                            f"""<div style="direction: rtl; text-align: right; background-color: #f0f2f6; padding: 20px; border-radius: 10px;">{answer}</div>""",
                            unsafe_allow_html=True)

                        with st.expander("منابع استفاده شده"):
                            for line in references.split("\n"):
                                st.markdown(f"""<div style="direction: rtl; text-align: right;">{line}</div>""",
                                            unsafe_allow_html=True)

        elif page == "فایل‌های پردازش شده":
            st.title("فایل‌های پردازش شده")

            # Add a session state flag for refreshing
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