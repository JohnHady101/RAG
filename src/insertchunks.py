from src.doc_loaders import pdf_loader
from src.pgvector_try import add_document


def insert_documents_from_pdf(pdf_path: str):
    documents = pdf_loader(pdf_path)
    for i, doc in enumerate(documents):
        if i <= 10:
            add_document(doc.page_content, metadata=doc.metadata)


PDF_PATH = "src/annualreport-2025.pdf"
insert_documents_from_pdf(PDF_PATH)
# documents = pdf_loader(pdf_path) 

# print(documents[0].metadata)