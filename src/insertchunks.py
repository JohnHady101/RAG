from doc_loaders import pdf_loader
from pgvector_try import add_document, search


def insert_documents_from_pdf(pdf_path: str):
    documents = pdf_loader(pdf_path)
    for i, doc in enumerate(documents):
        if i <= 10:
            add_document(doc.page_content, metadata=doc.metadata)



if __name__ == "__main__":
    # PDF_PATH = "src/annualreport-2025.pdf"
    # insert_documents_from_pdf(PDF_PATH)

    query= "by large number of government"
    results= search(query, top_k=7)

    for row in results:
        doc_id, content, metadata, similarity = row
        
        # Format the similarity score to 4 decimal places (e.g., 0.8934)
        # Multiply by 100 if you want a percentage (e.g., 89.34%)
        score_pct = similarity * 100 
        
        print(f"{score_pct:05.2f}%  | {content[:70]}...")


# documents = pdf_loader(pdf_path) 

# print(documents[0].metadata)