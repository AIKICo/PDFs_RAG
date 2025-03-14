import os
from typing import List, Union

from docling.document_converter import DocumentConverter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from TextProcessing.FastPersianNormalizer import FastPersianNormalizer


class EnhancedDoclingLoader(BaseLoader):
    def __init__(self, file_path: Union[str, List[str]]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()
        self.normalizer = FastPersianNormalizer()

    def load(self) -> List[LCDocument]:
        docs = []
        for source in self._file_paths:
            try:
                file_ext = os.path.splitext(source)[1].lower()
                if file_ext in ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.md', '.rtf', '.odt', '.ods', '.odp']:
                    dl_doc = self._converter.convert(source).document
                    text = dl_doc.export_to_markdown()
                    normalized_text = self.normalizer.normalize(text)
                    metadata = {"source": source, "filename": os.path.basename(source), "filetype": file_ext[1:]}
                    docs.append(LCDocument(page_content=normalized_text, metadata=metadata))
                else:
                    print(f"Unsupported file type: {file_ext}")
            except Exception as e:
                print(f"Error processing {source}: {e}")
        return docs
