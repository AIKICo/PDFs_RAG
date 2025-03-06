import os
import tempfile

import requests
import streamlit as st

from core.config import settings
from pdf.processor import PDFProcessor

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
    /* Other RTL styles remain the same */
</style>
""", unsafe_allow_html=True)


class StreamlitUI:
    def __init__(self):
        self.api_url = f"http://localhost:8000{settings.API_V1_STR}"
        self.token = None
        self.processor = PDFProcessor()

    def save_uploaded_file(self, uploaded_file):
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

    def login_page(self):
        """Display the login page."""
        st.title("ورود به سیستم")

        with st.form("login_form"):
            username = st.text_input("نام کاربری:")
            password = st.text_input("رمز عبور:", type="password")
            submitted = st.form_submit_button("ورود")

            if submitted:
                success, msg = self.login(username, password)
                if success:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error(msg)

                # Option to register
                st.write("حساب کاربری ندارید؟")
                if st.button("ثبت نام"):
                    st.session_state["show_page"] = "register"
                    st.rerun()

    def register_page(self):
        """Display the registration page."""
        st.title("ثبت نام کاربر جدید")

        with st.form("register_form"):
            username = st.text_input("نام کاربری:")
            email = st.text_input("ایمیل:")
            password = st.text_input("رمز عبور:", type="password")
            password_confirm = st.text_input("تایید رمز عبور:", type="password")
            submitted = st.form_submit_button("ثبت نام")

            if submitted:
                if password != password_confirm:
                    st.error("رمز عبور و تایید آن مطابقت ندارند.")
                else:
                    success, msg = self.register(username, email, password)
                    if success:
                        st.success("ثبت نام با موفقیت انجام شد. اکنون می‌توانید وارد شوید.")
                        st.session_state["show_page"] = "login"
                        st.rerun()
                    else:
                        st.error(msg)

        # Option to go back to login
        if st.button("بازگشت به صفحه ورود"):
            st.session_state["show_page"] = "login"
            st.rerun()

    def login(self, username: str, password: str) -> tuple:
        """
        Attempt to log in the user.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            response = requests.post(
                f"{self.api_url}/auth/token",
                data={"username": username, "password": password}
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state["token"] = data["access_token"]
                return True, "ورود موفقیت‌آمیز"
            else:
                return False, "نام کاربری یا رمز عبور نادرست است."
        except Exception as e:
            return False, f"خطا در اتصال به سرور: {e}"

    def register(self, username: str, email: str, password: str) -> tuple:
        """
        Register a new user.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            response = requests.post(
                f"{self.api_url}/auth/register",
                json={"username": username, "email": email, "password": password}
            )

            if response.status_code == 200:
                return True, "ثبت نام موفقیت‌آمیز"
            else:
                data = response.json()
                return False, data.get("detail", "خطا در ثبت نام")
        except Exception as e:
            return False, f"خطا در اتصال به سرور: {e}"

    def logout(self):
        """Log out the user."""
        if st.button("خروج", key="logout_button"):
            if "token" in st.session_state:
                del st.session_state["token"]
            if "logged_in" in st.session_state:
                del st.session_state["logged_in"]
            if "username" in st.session_state:
                del st.session_state["username"]
            st.rerun()

    def create_api_key(self, name: str, expires_in_days: int) -> tuple:
        """
        Create a new API key.

        Returns:
            Tuple of (success: bool, result: dict or str)
        """
        try:
            headers = {"Authorization": f"Bearer {st.session_state.get('token')}"}
            response = requests.post(
                f"{self.api_url}/auth/api-keys",
                json={"name": name, "expires_in_days": expires_in_days},
                headers=headers
            )

            if response.status_code == 200:
                return True, response.json()
            else:
                data = response.json()
                return False, data.get("detail", "خطا در ایجاد کلید API")
        except Exception as e:
            return False, f"خطا در اتصال به سرور: {e}"

    def main_page(self):
        """Display the main application page."""
        st.title("سیستم پرسش و پاسخ بر اساس فایل‌های PDF")

        # Show username in the sidebar
        st.sidebar.write(f"کاربر: {st.session_state.get('username', '')}")

        # Add logout button to sidebar
        self.logout()

        # Create tabs for different functions
        tab1, tab2, tab3, tab4 = st.tabs([
            "آپلود و پردازش فایل",
            "پرسش و پاسخ",
            "فایل‌های پردازش شده",
            "مدیریت کلیدهای API"
        ])

        with tab1:
            self.upload_tab()

        with tab2:
            self.query_tab()

        with tab3:
            self.files_tab()

        with tab4:
            self.api_keys_tab()

    def upload_tab(self):
        """Display the upload tab content."""
        st.header("آپلود و پردازش فایل‌های PDF")

        uploaded_files = st.file_uploader(
            "فایل‌های PDF را انتخاب کنید",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files and st.button("پردازش فایل‌ها"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Save files to temporary location
            pdf_paths = []
            for uploaded_file in uploaded_files:
                file_path = self.save_uploaded_file(uploaded_file)
                if file_path:
                    pdf_paths.append(file_path)

            if pdf_paths:
                # Define a progress callback
                def progress_callback(current, total, message=None):
                    if isinstance(total, int) and total > 0:
                        progress = min(current / total, 1.0)
                    else:
                        progress = current  # Assume current is already a fraction

                    progress_bar.progress(progress)
                    if message:
                        status_text.text(message)

                # Process PDFs
                result = self.processor.process_pdfs(pdf_paths, progress_callback)

                # Display results
                if result["errors"]:
                    st.error(f"خطاها: {', '.join(result['errors'])}")

                st.success(
                    f"تعداد {result['processed']} فایل پردازش شد. "
                    f"تعداد {result['skipped']} فایل قبلاً پردازش شده بود. "
                    f"تعداد {result['new_chunks']} قطعه جدید به پایگاه داده اضافه شد."
                )
            else:
                st.error("هیچ فایل PDF معتبری آپلود نشد.")

    def query_tab(self):
        """Display the query tab content."""
        st.header("پرسش و پاسخ")

        # Create a form for the query
        with st.form("query_form"):
            question = st.text_area("سوال خود را وارد کنید:")
            top_k = st.slider("تعداد منابع برای استخراج:", 1, 10, 4)
            submitted = st.form_submit_button("جستجو")

        if submitted and question:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Define a progress callback
            def progress_callback(progress, message=None):
                progress_bar.progress(progress)
                if message:
                    status_text.text(message)

            # Execute the query
            try:
                answer, sources = self.processor.query(question, top_k, progress_callback)

                # Display the answer
                st.subheader("پاسخ:")
                st.write(answer)

                # Display the sources
                if sources:
                    st.subheader("منابع:")
                    for source in sources:
                        with st.expander(f"منبع {source['id']}: {source['file']} (صفحه {source['page']})"):
                            st.write(source['content'])
            except Exception as e:
                st.error(f"خطا در اجرای جستجو: {e}")

    def files_tab(self):
        """Display the files tab content."""
        st.header("فایل‌های پردازش شده")

        if st.button("بارگزاری مجدد", key="reload_files"):
            st.session_state["reload_files"] = True

        # Get processed files
        files = self.processor.list_processed_files()

        if not files:
            st.info("هیچ فایلی پردازش نشده است.")
        else:
            # Display files in a table
            file_data = []
            for file in files:
                file_data.append({
                    "نام فایل": file["file_name"],
                    "تعداد صفحات": file["page_count"],
                    "تاریخ پردازش": file["processed_at"]
                })

            st.table(file_data)

    def api_keys_tab(self):
        """Display the API keys tab content."""
        st.header("مدیریت کلیدهای API")

        with st.form("api_key_form"):
            name = st.text_input("نام کلید (برای شناسایی):")
            expires_in_days = st.slider("مدت اعتبار (روز):", 1, 365, 30)
            submitted = st.form_submit_button("ایجاد کلید API جدید")

        if submitted and name:
            success, result = self.create_api_key(name, expires_in_days)
            if success:
                st.success("کلید API با موفقیت ایجاد شد.")

                # Display the API key
                st.code(result["api_key"], language=None)
                st.warning(
                    "این کلید را در جای امنی ذخیره کنید! "
                    "این تنها باری است که این کلید به شما نمایش داده می‌شود."
                )

                # Display expiration
                st.info(f"این کلید تا {result['expires_at']} معتبر است.")
            else:
                st.error(f"خطا در ایجاد کلید API: {result}")

    def run(self):
        """Run the Streamlit app."""
        # Initialize session state
        if "show_page" not in st.session_state:
            st.session_state["show_page"] = "login"

        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False

        # Display the appropriate page
        if not st.session_state["logged_in"]:
            if st.session_state["show_page"] == "register":
                self.register_page()
            else:
                self.login_page()
        else:
            self.main_page()

    # Now let's implement the main entry point:


if __name__ == "__main__":
    ui = StreamlitUI()
    ui.run()
