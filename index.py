import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Initialize the Hugging Face embedding model locally
hf_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Initialized HuggingFaceEmbeddings model.")



# Custom function to read PDFs from a folder
def read_pdfs_from_folder(folder_path: str):
    documents = []
    print(f"Reading PDFs from folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            print(f"Loaded {len(loaded_docs)} documents from {filename}.")
            documents.extend(loaded_docs)
    return documents

# Custom function to build and return the FAISS vector store
def build_faiss_vector_store(documents):
    print(f"Splitting {len(documents)} documents into smaller chunks.")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    print(f"Generated {len(split_docs)} chunks from documents.")
    faiss_store = FAISS.from_documents(split_docs, hf_embedding_model)
    print("Built FAISS vector store with the documents.")
    return faiss_store

# Accuracy test function
def accuracy_test(faiss_store):
    test_queries = {
       "Child labor in Nepal": "Child Migrants In Child Labour An Invisible Group In Need Of Attention.pdf",
        "Climate change impact in agriculture": "Impact Of Climate Change Finance In Agriculture On The Poor.pdf",
        "Disaster risk management in Udaypur": "Disaster Risk Management Plan Udaypur District (April 2011).pdf",
        "Food security in Nepal": "Food And Agriculture The Future Of Sustainability.pdf",
        "Evaluation of climate effects on diseases in Nepal": "Evaluation Of The Effects Of Climatic Factors On The Occurrence Of Diarrhoeal Diseases And Malaria A Pilot Retrospective Study In Jhapa District, Nepal - Technical Report.pdf"
    }

    correct_count = 0

    for query, expected_filename in test_queries.items():
        query_embedding = hf_embedding_model.embed_query(query)
        docs_and_scores = faiss_store.similarity_search_with_score_by_vector(query_embedding, top_k=1)
        
        if docs_and_scores:
            retrieved_metadata = docs_and_scores[0][0].metadata
            retrieved_filename = retrieved_metadata['source']
            
            # Compare the full filename
            if expected_filename in retrieved_filename:
                correct_count += 1
                print(f"Query: '{query}' - Passed (Expected: {expected_filename}, Got: {retrieved_filename})")
            else:
                print(f"Query: '{query}' - Failed (Expected: {expected_filename}, Got: {retrieved_filename})")
        else:
            print(f"Query: '{query}' - Failed (No documents retrieved)")

    total_queries = len(test_queries)
    accuracy = correct_count / total_queries if total_queries > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    cwd = os.getcwd()
    folder_path = f"{cwd}/data"  # Update this with your folder path
    documents = read_pdfs_from_folder(folder_path)
    vector_store = build_faiss_vector_store(documents)

    faiss_index_path = "vectorstore"
    vector_store.save_local(faiss_index_path)
    print(f"Vector store saved to {faiss_index_path}.")

    # Run the accuracy test
    accuracy_test(vector_store)