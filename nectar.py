# ============================================================
#  RAG Chatbot — mixedbread-ai Embeddings + Qwen2.5 0.5B
#  100% locale, nessuna API key necessaria
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# ============================================================
#  CONFIGURAZIONE
# ============================================================

LLM_MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Dispositivo: {DEVICE}")


# ============================================================
#  LLM — Qwen2.5 0.5B Instruct (scaricato automaticamente)
# ============================================================

def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=pipe)


# ============================================================
#  EMBEDDING MODEL — mixedbread-ai (scaricato automaticamente)
# ============================================================

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "Represent this sentence for searching relevant passages: ",
        },
    )


# ============================================================
#  DOCUMENT LOADER
# ============================================================

def document_loader(file):
    loader = PyPDFLoader(file)
    return loader.load()


# ============================================================
#  TEXT SPLITTER
# ============================================================

def split_text(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_documents(data)


# ============================================================
#  VECTOR DATABASE
# ============================================================

def vector_database(chunks):
    embedding_model = get_embedding_model()
    return Chroma.from_documents(chunks, embedding_model)


# ============================================================
#  RETRIEVER
# ============================================================

def build_retriever(file):
    splits = document_loader(file)
    chunks = split_text(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever(search_kwargs={"k": 3})


# ============================================================
#  PROMPT — formato ChatML nativo di Qwen
# ============================================================

def build_prompt():
    template = """<|im_start|>system
You are a helpful assistant. Answer the question using only the provided context. If the answer is not in the context, say "I don't have enough information to answer this question."<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


# ============================================================
#  HELPER
# ============================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
#  QA CHAIN
# ============================================================

def retriever_qa(file, query):
    try:
        if file is None:
            return "❌ Nessun file caricato. Carica un PDF prima di fare una domanda."
        if not query or query.strip() == "":
            return "❌ Inserisci una domanda."

        print(f"[DEBUG] File: {file}")
        print(f"[DEBUG] Query: {query}")

        llm = get_llm()
        print("[DEBUG] LLM pronto")

        retriever_obj = build_retriever(file)
        print("[DEBUG] Retriever pronto")

        prompt = build_prompt()

        chain = (
            {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(query)
        print("[DEBUG] Risposta generata")

        # Pulizia token residui di Qwen
        if "<|im_end|>" in result:
            result = result.split("<|im_end|>")[0]

        return result.strip()

    except Exception as e:
        import traceback
        print(f"[ERROR]\n{traceback.format_exc()}")
        return f"❌ Errore: {str(e)}"


# ============================================================
#  GRADIO INTERFACE
# ============================================================

rag_application = gr.Interface(
    fn=retriever_qa,
    #allow_flagging="never",
    inputs=[
        gr.File(
            label="📄 Carica PDF",
            file_count="single",
            file_types=['.pdf'],
            type="filepath",
        ),
        gr.Textbox(
            label="💬 Domanda",
            lines=2,
            placeholder="Es: Di cosa parla questo documento?",
        ),
    ],
    outputs=gr.Textbox(label="📝 Risposta", lines=8),
    title="🤖 RAG Chatbot — Qwen2.5 0.5B + mixedbread-ai",
    description="100% locale | Embeddings: mxbai-embed-large-v1 | LLM: Qwen2.5-0.5B-Instruct",
)

rag_application.launch(server_name='0.0.0.0', server_port=7860, share=True)