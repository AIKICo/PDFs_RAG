import os
import tempfile

import dash
import dash_bootstrap_components as dbc
import torch
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback

from DocumentProcess.DocumentProcess import DocumentProcess  # فرض می‌کنیم این ماژول موجود است

# تنظیمات اولیه محیط
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
torch.cuda.is_available = lambda: False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# راه‌اندازی برنامه Dash با Bootstrap برای استایل بهتر
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,
                                           "https://cdn.jsdelivr.net/npm/vazirmatn@33.0.3/Vazirmatn-font-face.css"])

# استایل‌های RTL و فونت Vazirmatn
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>سیستم RAG اسناد</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://cdn.jsdelivr.net/npm/vazirmatn@33.0.3/Vazirmatn-font-face.css');
            html, body, .app-container {
                direction: rtl;
                font-family: 'Vazirmatn', sans-serif !important;
                text-align: right;
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
            .sidebar {
                position: fixed;
                width: 250px;
                height: 100%;
                background-color: #f8f9fa;
                padding: 20px;
                direction: rtl;
                text-align: right;
            }
            .content {
                margin-right: 270px;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# آیکون‌ها برای نوع فایل
def get_file_icon(file_type):
    icons = {
        'pdf': '📄', 'docx': '📝', 'doc': '📝', 'xlsx': '📊', 'xls': '📊',
        'pptx': '📊', 'ppt': '📊', 'txt': '📋', 'md': '📝', 'rtf': '📄',
        'odt': '📝', 'ods': '📊', 'odp': '📊',
    }
    return icons.get(file_type.lower(), '📁')


# ذخیره‌سازی فایل آپلود شده
def save_uploaded_file(uploaded_file):
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
    except Exception as e:
        return None


# تعریف ساختار رابط کاربری
app.layout = html.Div([
    # نوار کناری
    html.Div([
        html.H3("سیستم RAG اسناد"),
        html.H5("با پشتیبانی از زبان فارسی"),
        dcc.RadioItems(
            id="page-selector",
            options=[
                {"label": "پردازش فایل‌ها", "value": "process"},
                {"label": "پرسش و پاسخ", "value": "qa"},
                {"label": "فایل‌های پردازش شده", "value": "processed"},
            ],
            value="process",
            style={"margin-top": "20px"}
        ),
        html.Hr(),
        html.H5("تنظیمات مدل"),
        dcc.Input(id="model-name", type="text", value="gemma3", placeholder="مدل LLM"),
        dcc.Input(id="embeddings-model", type="text", value="intfloat/multilingual-e5-large",
                  placeholder="مدل Embeddings"),
    ], className="sidebar"),

    # محتوای اصلی
    html.Div(id="page-content", className="content")
])


# تعریف محتوای پویا برای هر صفحه
@callback(
    Output("page-content", "children"),
    Input("page-selector", "value"),
    State("model-name", "value"),
    State("embeddings-model", "value")
)
def render_page_content(page, model_name, embeddings_model):
    processor = DocumentProcess(model_name=model_name, embeddings_model=embeddings_model)

    if page == "process":
        return html.Div([
            html.H1("پردازش اسناد"),
            html.P("فایل‌های خود را آپلود کنید تا پردازش شوند."),
            dcc.Upload(
                id="upload-data",
                children=html.Button("انتخاب فایل‌ها"),
                multiple=True
            ),
            html.Button("پردازش فایل‌ها", id="process-button", n_clicks=0, className="btn btn-primary mt-3"),
            html.Div(id="process-output")
        ])

    elif page == "qa":
        return html.Div([
            html.H1("پرسش و پاسخ از اسناد"),
            html.Div(id="chat-history"),
            dcc.Textarea(id="query-input", placeholder="پرسش خود را وارد کنید:",
                         style={"width": "100%", "height": "100px"}),
            dbc.Row([
                dbc.Col(dcc.Input(id="top-k", type="number", value=4, min=1, max=10, step=1), width=3),
                dbc.Col(html.Button("ارسال", id="submit-query", n_clicks=0, className="btn btn-primary"), width=2),
                dbc.Col(html.Button("پاک کردن گفتگو", id="clear-chat", n_clicks=0, className="btn btn-danger"),
                        width=2),
            ]),
            html.Div(id="query-output")
        ])

    elif page == "processed":
        files = processor.list_processed_files()
        data = [
            {
                "شماره": i + 1,
                "نوع": f"{get_file_icon(file.get('file_type', 'نامشخص'))} {file.get('file_type', 'نامشخص')}",
                "نام فایل": file["file_name"],
                "اندازه": f"{file.get('metadata', {}).get('file_size_mb', '')} MB" if file.get('metadata', {}).get(
                    'file_size_mb', '') else "",
                "تعداد صفحات": file["page_count"],
                "تاریخ پردازش": file["processed_at"],
                "عملیات": processor._calculate_file_hash(file["file_path"]) if file.get("file_path") and os.path.exists(
                    file["file_path"]) else ""
            } for i, file in enumerate(files)
        ]
        return html.Div([
            html.H1("فایل‌های پردازش شده"),
            html.Button("🔄 تازه‌سازی", id="refresh-files", n_clicks=0, className="btn btn-info mb-3"),
            dash_table.DataTable(
                id="processed-files-table",
                columns=[
                    {"name": "شماره", "id": "شماره"},
                    {"name": "نوع", "id": "نوع"},
                    {"name": "نام فایل", "id": "نام فایل"},
                    {"name": "اندازه", "id": "اندازه"},
                    {"name": "تعداد صفحات", "id": "تعداد صفحات"},
                    {"name": "تاریخ پردازش", "id": "تاریخ پردازش"},
                    {"name": "عملیات", "id": "عملیات", "presentation": "markdown"}
                ],
                data=data,
                style_table={"direction": "rtl", "textAlign": "right"}
            ),
            html.Div(id="delete-output")
        ])


# Callback برای پردازش فایل‌ها
@callback(
    Output("process-output", "children"),
    Input("process-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("model-name", "value"),
    State("embeddings-model", "value"),
    prevent_initial_call=True
)
def process_files(n_clicks, contents, filenames, model_name, embeddings_model):
    if not contents:
        return html.P("لطفاً ابتدا فایل‌ها را آپلود کنید.")

    processor = DocumentProcess(model_name=model_name, embeddings_model=embeddings_model)
    file_paths = [save_uploaded_file(content) for content in contents]
    file_paths = [fp for fp in file_paths if fp is not None]

    if file_paths:
        result = processor.process_documents(file_paths,
                                             lambda current, total, msg: None)  # بدون آپدیت پیشرفت در این مثال
        return html.Div([
            dbc.Row([
                dbc.Col(html.P(f"فایل‌های پردازش شده: {result['processed']}"), width=4),
                dbc.Col(html.P(f"فایل‌های رد شده: {result['skipped']}"), width=4),
                dbc.Col(html.P(f"تکه‌های متنی جدید: {result['new_chunks']}"), width=4),
            ]),
            html.P(f"{result['processed']} فایل با موفقیت پردازش شد.", className="text-success")
        ])
    return html.P("خطا در پردازش فایل‌ها.")


# Callback برای پرسش و پاسخ
@callback(
    [Output("chat-history", "children"), Output("query-output", "children")],
    [Input("submit-query", "n_clicks"), Input("clear-chat", "n_clicks")],
    [State("query-input", "value"), State("top-k", "value"), State("model-name", "value"),
     State("embeddings-model", "value")],
    prevent_initial_call=True
)
def update_chat(submit_clicks, clear_clicks, query, top_k, model_name, embeddings_model):
    ctx = dash.callback_context
    processor = DocumentProcess(model_name=model_name, embeddings_model=embeddings_model)

    if not hasattr(app, "chat_history"):
        app.chat_history = []

    if ctx.triggered_id == "clear-chat":
        app.chat_history = []
        processor.clearChatHitsory()
        return [], html.P("گفتگو پاک شد.", className="text-success")

    if query:
        app.chat_history.append({"role": "user", "content": query})
        response = list(processor.query(query, top_k, lambda p, m: None))[-1]  # آخرین پاسخ از استریم
        app.chat_history.append({"role": "assistant", "content": response})

    chat_content = [
        html.Div(f"👤 {msg['content']}", className="chat-message user-message") if msg["role"] == "user" else
        html.Div([
            html.Div(f"🤖 {msg['content']['answer']}", className="chat-message assistant-message"),
            html.Button(f"📚 نمایش منابع", id={"type": "show-sources", "index": i}, className="btn btn-sm btn-info"),
            dcc.Store(id={"type": "sources-store", "index": i}, data=msg["content"]["sources"])
        ], style={"marginBottom": "10px"})
        for i, msg in enumerate(app.chat_history)
    ]
    return chat_content, html.P("پاسخ با موفقیت دریافت شد.", className="text-success")


# اجرای برنامه
if __name__ == "__main__":
    app.run(debug=True)