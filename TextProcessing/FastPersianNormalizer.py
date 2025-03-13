import re
from functools import lru_cache

import spacy


class FastPersianNormalizer:
    def __init__(self):
        """
        نرمال‌ساز سریع متن فارسی با استفاده از spaCy و regex
        """
        # بارگذاری مدل زبانی کوچک و سریع
        # برای نصب: python -m spacy download xx_sent_ud_sm
        try:
            self.nlp = spacy.load("xx_sent_ud_sm", disable=["ner", "parser", "tagger"])
        except OSError:
            # اگر مدل نصب نشده باشد، از فقط از regex استفاده می‌کنیم
            self.nlp = None

        # الگوهای regex برای نرمال‌سازی متن فارسی
        self.patterns = [
            (re.compile(r'[ـ\r]'), ''),  # حذف کشیده و کاراکتر CR
            (re.compile(r' +'), ' '),  # تبدیل چندین فاصله به یک فاصله
            (re.compile(r'[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652]'), ''),  # حذف اعراب
            (re.compile(r'ى'), 'ی'),  # تبدیل ی عربی به ی فارسی
            (re.compile(r'ي'), 'ی'),  # تبدیل ی عربی به ی فارسی
            (re.compile(r'ك'), 'ک'),  # تبدیل ک عربی به ک فارسی
            (re.compile(r'ة'), 'ه'),  # تبدیل ة به ه
            (re.compile(r"[إأآا]"), 'ا'),  # یکسان‌سازی الف‌ها
        ]

    @lru_cache(maxsize=10000)
    def normalize(self, text):
        """
        نرمال‌سازی متن فارسی با استفاده از spaCy (اگر موجود باشد) و regex

        Args:
            text: متن ورودی برای نرمال‌سازی

        Returns:
            متن نرمال‌سازی شده
        """
        # اعمال پردازش spaCy اگر موجود باشد
        if self.nlp is not None:
            # محدود کردن طول متن برای جلوگیری از خطای متن طولانی
            if len(text) > 1000000:  # حداکثر طول متن
                chunks = [text[i:i + 1000000] for i in range(0, len(text), 1000000)]
                normalized_chunks = []
                for chunk in chunks:
                    doc = self.nlp(chunk)
                    normalized_chunks.append(doc.text)
                text = ''.join(normalized_chunks)
            else:
                doc = self.nlp(text)
                text = doc.text

        # اعمال الگوهای regex
        for pattern, repl in self.patterns:
            text = pattern.sub(repl, text)

        # نهایی‌سازی متن
        text = text.strip()

        return text