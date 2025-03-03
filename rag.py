from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import os
import torch
import requests
import hashlib
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER = "pdf"
#EMBEDDING_MODEL = "model/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "model_trained/wangchanberta"
VECTOR_STORE_PATH = "vector_store"
METADATA_PATH = VECTOR_STORE_PATH + "index.pkl"
"""
web_urls = [
    "https://www.tru.ac.th/",
    "https://www.google.com/",
    "https://th.wikipedia.org/wiki/%E0%B8%A1%E0%B8%AB%E0%B8%B2%E0%B8%A7%E0%B8%B4%E0%B8%97%E0%B8%A2%E0%B8%B2%E0%B8%A5%E0%B8%B1%E0%B8%A2%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%A0%E0%B8%B1%E0%B8%8F%E0%B9%80%E0%B8%97%E0%B8%9E%E0%B8%AA%E0%B8%95%E0%B8%A3%E0%B8%B5"
]"""

def get_file_hash(file_path):
    """คำนวณ hash ของไฟล์เพื่อตรวจสอบการเปลี่ยนแปลง"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_existing_metadata(METADATA_PATH):
    """โหลดข้อมูลไฟล์ PDF ที่เคยแปลงเป็นเวกเตอร์แล้ว"""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)

    return {}

def save_metadata(metadata, METADATA_PATH):
    """บันทึกข้อมูลไฟล์ PDF ที่แปลงแล้ว"""
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def create_vector_store(folder_path, web_urls=None):
    """สร้างเวกเตอร์ฐานข้อมูลจาก PDF + เว็บ พร้อม metadata"""
    documents = []
    metadata_dict = {}  # เก็บข้อมูล metadata
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"📂 กำลังโหลด: {pdf_path}")
            
            loader = PDFPlumberLoader(pdf_path)
            pdf_docs = loader.load()
            
            # เพิ่ม Metadata
            for doc in pdf_docs:
                doc.metadata = {"filename": filename}
            
            documents.extend(pdf_docs)

    print(f"✅ โหลดข้อมูลสำเร็จ ({len(documents)} ไฟล์)")

    # โหลดข้อมูลจากเว็บ
    if web_urls:
        for url in web_urls:
            print(f"🌐 กำลังดึงข้อมูลจากเว็บ: {url}")
            web_doc = fetch_web_data(url)
            if web_doc:
                documents.append(web_doc)
                print(f"✅ โหลดข้อมูลจาก {url} สำเร็จ")
            else:
                print(f"⚠️ ดึงข้อมูลจาก {url} ไม่สำเร็จ")

    # ตั้งค่า embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # แบ่งข้อความเป็น chunk พร้อม metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # บันทึก metadata
    for idx, doc in enumerate(docs):
        metadata_dict[idx] = doc.metadata

    print(f"📑 จำนวนเอกสารที่จะแปลงเป็นเวกเตอร์: {len(docs)}")

    # สร้าง FAISS vector database
    vector_db = FAISS.from_documents(docs, embedding_model)
    vector_db.save_local(VECTOR_STORE_PATH)

    # บันทึก Metadata
    save_metadata(metadata_dict, METADATA_PATH)
    print("✅ สร้างเวกเตอร์เสร็จแล้ว!")

def update_vector_store(PDF_FOLDER, embedding_model):
    """อัปเดต FAISS Vector Store โดยเพิ่มเฉพาะไฟล์ใหม่ พร้อม Metadata"""
    # โหลด metadata ของไฟล์ที่เคยสร้างเวกเตอร์
    metadata = load_existing_metadata(METADATA_PATH)

    new_docs = []
    new_metadata = {}

    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, filename)
            file_hash = get_file_hash(file_path)

            # ถ้าไฟล์ยังไม่เคยโหลดมาก่อน หรือมีการเปลี่ยนแปลง
            if filename not in metadata or metadata[filename] != file_hash:
                print(f"📌 พบไฟล์ใหม่หรือไฟล์ที่ถูกแก้ไข: {filename}")

                # โหลดข้อมูลจาก PDF
                loader = PDFPlumberLoader(file_path)
                documents = loader.load()

                # แบ่งข้อความเป็น chunk พร้อม metadata
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunked_docs = text_splitter.split_documents(documents)

                for doc in chunked_docs:
                    doc.metadata = {"filename": filename}  # ใส่ metadata
                    new_docs.append(doc)
                    new_metadata[len(metadata) + len(new_metadata)] = doc.metadata  # อัปเดต Metadata ใหม่

                # บันทึก hash ของไฟล์
                metadata[filename] = file_hash

    if new_docs:
        print("📌 อัปเดต FAISS โดยเพิ่มไฟล์ใหม่...")

        # โหลด FAISS เก่าหรือสร้างใหม่
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            vector_db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model)
        else:
            vector_db = FAISS.from_documents([], embedding_model)

        # เพิ่มเอกสารใหม่เข้า vector store
        vector_db.add_documents(new_docs)
        vector_db.save_local(VECTOR_STORE_PATH)

        # บันทึก Metadata
        metadata.update(new_metadata)
        save_metadata(metadata, METADATA_PATH)

        print("✅ อัปเดตเวกเตอร์สำเร็จ!")
    else:
        print("⚡ ไม่มีไฟล์ใหม่ ไม่ต้องอัปเดต FAISS")

def search_from_vector_store(query, top_k=3):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # โหลด FAISS
    vector_db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)

    # แปลงคำถามเป็นเวกเตอร์
    query_vector = embedding_model.embed_query(query)

    # ค้นหาข้อมูลที่ใกล้เคียง
    results = vector_db.similarity_search_by_vector(query_vector, k=top_k)

    return results


def fetch_web_data(url):
    """ดึงข้อมูลจากเว็บเพจ"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # เอาเฉพาะข้อความที่ต้องการจากเว็บ
    text = soup.get_text(separator="\n", strip=True)
    
    return Document(page_content=text, metadata={"source": url})

def search_query(query, k=3):
    # โหลด FAISS vector database
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)

    # ค้นหาข้อมูล
    retrieved_docs = vector_db.similarity_search(query, k=k)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
    print("retrieved_docs", retrieved_docs)
    return retrieved_text


def loadtyphoon2(modelconfig):
    load_path = os.path.join("model_trained", modelconfig)
    #load_path = "scb10x/llama3.2-typhoon2-1b-instruct"
    # Reload the saved quantized model
    typhoon2_model = AutoModelForCausalLM.from_pretrained(
        load_path,
        device_map="auto",
        local_files_only=True
    )
    typhoon2_tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
    return typhoon2_model, typhoon2_tokenizer

def typhoon2chat(query, typhoon2_model, typhoon2_tokenizer):
    messages = [
        {"role": "system", "content": "You are a male AI assistant named Typhoon created by SCB 10X to be helpful, harmless, and honest. Typhoon is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks. Typhoon responds directly to all human messages without unnecessary affirmations or filler phrases like “Certainly!”, “Of course!”, “Absolutely!”, “Great!”, “Sure!”, etc. Specifically, Typhoon avoids starting responses with the word “Certainly” in any way. Typhoon follows this information in all languages, and always responds to the user in the language they use or request. Typhoon is now being connected with a human. Write in fluid, conversational prose, Show genuine interest in understanding requests, Express appropriate emotions and empathy. Also showing information in term that is easy to understand and visualized."},
        {"role": "user", "content": query},
    ]

    input_ids = typhoon2_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    terminators = [
        typhoon2_tokenizer.eos_token_id,
        typhoon2_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    ## Typhoon 1b need low temperature to inference.
    outputs = typhoon2_model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.8,
    )

    response = outputs[0][input_ids.shape[-1]:]
    #print("Token decoded",typhoon2_tokenizer.decode(response, skip_special_tokens=True))
    return typhoon2_tokenizer.decode(response, skip_special_tokens=True)

# สร้างเวกเตอร์ฐานข้อมูลครั้งแรก (รันเพียงครั้งเดียว หรือเมื่อมีการเพิ่มไฟล์)
if not os.path.exists(VECTOR_STORE_PATH):
    create_vector_store(FOLDER)

#prompts = f"""
#   {retrieved_text}
#   Q:
#   {query}
#   A:
#"""

"""
query = "ออกข้อสอบ คำถาม + เฉลย ของมหาวิทยาลัยราชภัฏเทพสตรีให้หน่อยครับ"

#retrieved_text = search_query(query, k=3)
retrieved_text = search_from_vector_store(query=query, top_k=3)

print("ข้อมูลที่พบ:\n", retrieved_text)



modelconfig = "local_typhoon2_1b"
typhoon2_model, typhoon2_tokenizer = loadtyphoon2(modelconfig)
response = typhoon2chat(prompts, typhoon2_model, typhoon2_tokenizer)

print(f"typhoon2 response: {response}")
"""