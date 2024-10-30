from flask import Flask, Response, request
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator
import json
import time
from typing import Iterable, Literal
from uuid import uuid4
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Flask app setup
system_fingerprint = str(uuid4())
app = Flask(__name__)
model = "llama2"
llm = ChatOllama(model=model, max_tokens=100)
print("test1")

# Load the FAISS vector store created in index.py
faiss_index_path = "vectorstore"
hf_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(faiss_index_path, hf_embedding_model, allow_dangerous_deserialization=True)
print("Loaded FAISS vector store.")

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class OpenAIRequest(BaseModel):
    model: str
    temperature: float
    top_p: float = Field(gt=0)
    presence_penalty: float
    frequency_penalty: float
    user: str
    stream: bool
    messages: list[Message]

@RunnableGenerator
def stream_openai_response(chunks: Iterable[AIMessageChunk]) -> Iterable[bytes]:
    for chunk in chunks:
        data = json.dumps({
            "id": chunk.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.content},
                    "logprobs": None,
                    "finish_reason": None
                }
            ]
        })

        b = bytes(f"data: {data}\n\n", "utf-8")
        print(f"Streaming chunk: {chunk.content}")
        yield b

# def build_chain():
#     template = """
#     ###System:
#     You are the Assistant for UN documents provided in context. 
#     You have to answer the user's questions using only the context provided to you. List the source file in bullet form. Also mention keywords for the answers like Keywords :example, example, etc. If you don't know the answer, \
#     just say you don't know. 

#     Be strict, Don't try to make up an answer. 

#     if provided {context} == "missing": response I don't know, Don't try to make up an answer. 
#     ###Context:
#     {context}

#     ### User:
#     {query}
#     ###Response:
#     Also print reference of source as {metadata}
#     """

#     prompt = ChatPromptTemplate.from_template(template)
#     print("Built chat prompt template.............", prompt)
#     return prompt | llm

def build_chain():
    template = """
    ###System:
    You are the Assistant for UN documents provided in context. 
    You have to answer the user's questions using only the context provided to you. List the source file in bullet form. Also mention keywords for the answers like Keywords: example, example, etc. If you don't know the answer, \
    just say you don't know. 

    Be strict, Don't try to make up an answer. 

    if provided {context} == "missing": response I don't know, Don't try to make up an answer. 
    ###Context:
    {context}

    ### User:
    {query}
    ###Response:
    Also print reference of source as {metadata}
    """

    prompt = ChatPromptTemplate.from_template(template)
    print("Built chat prompt template.............", prompt)
    return prompt | llm



chain = build_chain() | stream_openai_response
print("Initialized the LangChain chain.")

@app.route("/chat/completions/", methods=["POST"])
def chat():
    print("Received a request at /chat/completions")
    chat = OpenAIRequest(**request.get_json())
    question = chat.messages[-1].content
    print(f"Received question: {question}")

    query_embedding = hf_embedding_model.embed_query(question)
    print(f"Generated embedding for the query. Query embedding: {query_embedding}")

    docs_and_scores = vector_store.similarity_search_with_score_by_vector(query_embedding, top_k=10, score_threshold=1.3)
    print(f"Retrieved {len(docs_and_scores)} documents with scores.", docs_and_scores)

    docs = [doc for doc, _ in docs_and_scores]

    if docs:
        context = " ".join([doc.page_content for doc in docs])
        metadata = "\n".join([f"- {doc.metadata['source']} (Page {doc.metadata['page']})" for doc in docs])
        print("Generated context from filtered documents.", docs[0].metadata)
    else:
        context = "missing"
        metadata = "empty"
        print("No relevant documents found; setting context to 'I don't know'.")


    # Include the final response in the chain input
    chain_input = {"query": question, "context": context, "metadata": metadata}

    return Response(chain.stream(chain_input), mimetype="text/event-stream")

    # # Include the final response in the chain input
    # chain_input = {"query": question, "context": context, "metadata": metadata}

    # # Generate the response from the chain
    # generated_response = "".join([chunk.decode("utf-8") for chunk in chain.stream(chain_input)])

    # # Append the metadata to the generated response
    # final_response_with_metadata = f"{generated_response}\n\n**Source(s):**\n{metadata}"

    # return Response(final_response_with_metadata, mimetype="text/event-stream")



if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True, port=8000)
