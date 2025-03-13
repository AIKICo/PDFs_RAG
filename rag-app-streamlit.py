import os
import tempfile
import torch
import streamlit as st
import torch
from langchain.chains import RetrievalQA

from FileProcessing.DocumentProcess import DocumentProcess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit configuration
st.set_page_config(
    page_title="سیستم RAG فایل‌های PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set RTL direction for the entire app
st.markdown("""
<style>
    .stApp {
        direction: rtl;
    }
    div[data-testid="stForm"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stMarkdown"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stText"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stTable"] th, div[data-testid="stTable"] td {
        text-align: right;
    }
    div[data-testid="stFileUploader"] > label {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label {
        direction: rtl;
        text-align: right;
    }
    button[kind="primary"], button[kind="secondary"] {
        float: right;
    }
    div[data-testid="stMetric"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    section[data-testid="stSidebar"] > div {
        direction: rtl;
    }
    div[data-testid="stExpander"] {
        direction: rtl;
    }
    div[data-testid="stSlider"] > div {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path."""
    try:
        # Create a temporary file path
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        st.error(f"خطا در ذخیره فایل: {e}")
        return None


def main():
    # Sidebar for settings and navigation
    st.sidebar.title("سیستم RAG فایل‌های PDF")
    st.sidebar.subheader("با پشتیبانی از زبان فارسی")

    # Navigation
    page = st.sidebar.radio("صفحات", ["پردازش فایل‌ها", "پرسش و پاسخ", "فایل‌های پردازش شده"])

    # Model settings
    st.sidebar.subheader("تنظیمات مدل")
    model_name = st.sidebar.text_input("مدل LLM (gemma3)", value="gemma3")
    embeddings_model = st.sidebar.text_input("مدل Embeddings", value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Initialize processor
    processor = DocumentProcess(model_name=model_name, embeddings_model=embeddings_model)

    try:
        # Process PDF files page
        if page == "پردازش فایل‌ها":
            st.title("پردازش فایل‌های PDF")
            st.write("فایل‌های PDF خود را آپلود کنید تا پردازش شوند.")

            uploaded_files = st.file_uploader("فایل‌های PDF را انتخاب کنید",
                                              type="pdf",
                                              accept_multiple_files=True)

            col1, col2 = st.columns([4, 1])
            with col2:
                process_button = st.button("پردازش فایل‌ها", type="primary")

            if uploaded_files and process_button:
                pdf_paths = []

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Save uploaded files to temp location
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        pdf_paths.append(file_path)

                if pdf_paths:
                    # Define callback for progress updates
                    def update_progress(current, total, message=""):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)

                    # Process PDFs with progress updates
                    result = processor.process_documents(pdf_paths, update_progress)

                    # Display results
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

        # Query page
        elif page == "پرسش و پاسخ":
            st.title("پرسش و پاسخ از اسناد")

            # Check if there are documents in the database
            files = processor.list_processed_files()
            if not files:
                st.warning("هیچ فایلی در پایگاه داده موجود نیست. لطفا ابتدا فایل‌ها را پردازش کنید.")
            else:
                st.write(f"{len(files)} فایل در پایگاه داده موجود است.")

                # Query input
                query = st.text_area("پرسش خود را وارد کنید:", height=100)
                top_k = st.slider("تعداد منابع برای بازیابی:", min_value=1, max_value=10, value=4)

                col1, col2 = st.columns([4, 1])
                with col2:
                    submit_button = st.button("ارسال پرسش", type="primary")

                if submit_button:
                    if not query:
                        st.warning("لطفا یک پرسش وارد کنید.")
                    else:
                        # Show spinners and progress
                        progress_placeholder = st.empty()
                        progress_bar = st.progress(0)

                        def update_query_progress(progress, message):
                            progress_bar.progress(progress)
                            progress_placeholder.text(message)

                        # Execute query
                        with st.spinner("در حال پردازش پرسش..."):
                            response = processor.query(query, top_k, update_query_progress)

                        # Display response
                        st.subheader("پاسخ:")

                        # Split answer and references
                        parts = response.split("\n\nمنابع:\n")
                        answer = parts[0].replace("پاسخ: ", "")
                        references = parts[1] if len(parts) > 1 else ""

                        st.markdown(f"""<div style="direction: rtl; text-align: right;">{answer}</div>""",
                                    unsafe_allow_html=True)

                        # Display references in an expander
                        with st.expander("منابع"):
                            for line in references.split("\n"):
                                st.markdown(f"""<div style="direction: rtl; text-align: right;">{line}</div>""",
                                            unsafe_allow_html=True)

        # List processed files page
        elif page == "فایل‌های پردازش شده":
            st.title("فایل‌های پردازش شده")

            files = processor.list_processed_files()
            if not files:
                st.info("هیچ فایلی پردازش نشده است.")
            else:
                st.write(f"{len(files)} فایل در پایگاه داده موجود است.")

                # Create a table to display files
                data = []
                for i, file in enumerate(files, 1):
                    data.append({
                        "شماره": i,
                        "نام فایل": file["file_name"],
                        "تعداد صفحات": file["page_count"],
                        "تاریخ پردازش": file["processed_at"]
                    })

                st.table(data)

    finally:
        processor.close()


if __name__ == "__main__":
    main()
