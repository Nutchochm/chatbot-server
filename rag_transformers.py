from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, pipeline
from pymongo import MongoClient
from PyPDF2 import PdfReader
from PIL import Image
from flask_socketio import SocketIO, emit
import os
import datetime
import time
import json
import uuid
import faiss
import fitz
import pickle
import docx
import re
import numpy as np
import torch
import pytesseract
import io
import threading

socketio = SocketIO()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name ="model_trained/wangchanberta"
# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Initialize FAISS index
d = 768  # Embedding dimension
#faiss_index = faiss.IndexFlatL2(d)
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(d))

vector_store = []  # Store metadata

def extract_text_from_pdf(pdf_path, file_id):
    try:
        doc = fitz.open(pdf_path)
        text_list = []
        for page in doc:
            text = page.get_text("text").encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            text_list.append(text)
        
        # เช็คว่าได้ข้อความหรือไม่
        if not any(text.strip() for text in text_list):
            print(f"❌ No text found in PDF {pdf_path}, trying OCR method...")
            # ถ้าไม่พบข้อความ, ใช้ OCR
            text_list = extract_text_ocr_pdf(pdf_path, file_id)
        
        return text_list
    except Exception as e:
        print(f"❌ Error extracting text from PDF {pdf_path}: {e}")
        return []

def extract_text_ocr_pdf(pdf_path, file_id):
    try:
        doc = fitz.open(pdf_path)
        text_list = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # ถ้าเป็นหน้า PDF ที่มีรูปภาพ
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))  # แปลงเป็นรูปภาพ PNG

            # ใช้ OCR ในการแปลงรูปภาพเป็นข้อความ
            text_from_image = pytesseract.image_to_string(img)

            # เช็คว่า OCR สามารถดึงข้อความออกมาได้หรือไม่
            if text_from_image.strip() == "":
                print(f"❌ No text found by OCR on page {page_num + 1} in PDF {pdf_path}")
                text_list.append("No text found by OCR")
            else:
                text_list.append(text_from_image)
        
        return text_list
    except Exception as e:
        print(f"❌ Error extracting text from PDF {pdf_path}: {e}")
        return []

def extract_text_from_txt(file_path, file_id):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return [text]

def extract_text_from_json(file_path, file_id):
    import json
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [json.dumps(data)]

def extract_text_from_docx(file_path, file_id):
    from docx import Document
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + '\n'
    return [text]

def get_tokenizer(model_name):
    """ดึง tokenizer สำหรับแต่ละ thread"""
    if not hasattr(thread_local, "tokenizer"):
        thread_local.tokenizer = AutoTokenizer.from_pretrained(model_name)
    return thread_local.tokenizer

def split_text(text, chunk_size=256):
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.tokenize(text)
    chunks = [tokenizer.convert_tokens_to_string(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def preprocess_query(query):
    query = query.lower().strip()
    query = re.sub(r"[^a-zA-Z0-9ก-๙\s]", "", query)
    return query


read_files = []

# ฟังก์ชันในการแปลงข้อความเป็น BERT embeddings
def extract_bert_embeddings(text):
    # แปลงข้อความให้เป็น token IDs ที่ BERT สามารถเข้าใจได้
    if isinstance(text, list):  
        text = " ".join(text) 
    tokenized = get_tokenizer(model_name)
    inputs = tokenized(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    #print(f"โมเดลรองรับ : {output}")
    #embeddings = output.last_hidden_state[:, 0, :].numpy()
    #return embeddings
    embeddings = output.last_hidden_state.mean(dim=1)  # ใช้ mean pooling
    return embeddings.numpy()

file_id = str(uuid.uuid4())

def get_next_file_number(output_path):
    """ หาหมายเลขไฟล์ถัดไปจาก vector_store """
    existing_files = [f for f in os.listdir(output_path) if f.endswith('.faiss')]
    file_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('file_') and f.split('_')[1].split('.')[0].isdigit()]
    return max(file_numbers, default=0) + 1



def insert_metadata(role, file_name, file_type, file_id, input_file_name): 
    try:
        connectionname = 'vectors_' + role 
        client = MongoClient('mongodb://localhost:27017')
        db = client['rag_db']
        collection = db[connectionname]


        metadata = {
            "file_name": file_name, 
            "file_type": file_type, 
            "file_id": file_id, 
            "input_file_name": input_file_name, 
            "created_at": datetime.datetime.utcnow() 
        }

        result = collection.update_one(
            {"file_id": file_id},
            {"$set": metadata},
            upsert=True
        )

        if result.upserted_id:
            print(f"New metadata inserted for file {file_name}")
        else:
            print(f"Metadata for file {file_name} already exists.")


    except Exception as e:
        print(f"Error inserting metadata into MongoDB: {e}")

def search_faiss_in_folder(query_text, faiss_folder, top_k=3):
    query_embedding = extract_bert_embeddings([query_text])
    #print(f"🔎 Query: {query_text}")
    #print("Query embedding shape before reshape:", query_embedding.shape)

    # ✅ Reshape ถ้าจำเป็น
    if len(query_embedding.shape) == 3:
        query_embedding = query_embedding.reshape(query_embedding.shape[1:])
    elif len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    #print("Query embedding shape after reshape:", query_embedding.shape)
    #print(f"🔍 ตัวอย่าง Query embedding: {query_embedding[:5]}")
    faiss_files = [f for f in os.listdir(faiss_folder) if f.endswith('.faiss')]
    pickle_files = [f for f in os.listdir(faiss_folder) if f.endswith('.pkl')]
    all_distances = []
    all_indices = []
    retrieved_texts = []
    metadata = []

    for faiss_file, pickle_file in zip(faiss_files, pickle_files):
        index_path = os.path.join(faiss_folder, faiss_file)
        pickle_path = os.path.join(faiss_folder, pickle_file)

        # ✅ โหลด FAISS index
        index = faiss.read_index(index_path)

        # ✅ โหลด pickle
        with open(pickle_path, "rb") as f:
            text_list = pickle.load(f)

        #print(f"📂 Searching in {faiss_file} (dimension: {index.d})")

        # ❌ ตรวจสอบว่าขนาดตรงกันหรือไม่
        if query_embedding.shape[1] != index.d:
            #raise ValueError(f"Dimension mismatch: query embedding has {query_embedding.shape[1]} dimensions, "
            #                 f"but index expects {index.d} dimensions.")
            print(f"⚠️ Skipping {faiss_file}: Dimension mismatch ({query_embedding.shape[1]} vs {index.d})")
            continue  
        
        # ✅ ค้นหา
        distances, indices = index.search(query_embedding, min(3, index.ntotal))
        #print(f"🎯 ค่าที่คืนมา: indices = {indices}, distances = {distances}")
        #print(f"🎯 FAISS Results ({faiss_file}):")
        """for idx in (indices[0]):
            if idx >= 0 and idx < len(text_list):
                retrieved_texts.append(text_list[idx]) # text_list to text_list_split
                metadata.append({
                    "file_name": pickle_file,
                    "distance": distances[0],
                    "text": text_list[idx]
                })"""
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >=0 and idx < len(text_list):
                retrieved_texts.append(text_list[idx])
                metadata.append({
                    "file_name": pickle_file,
                    "faiss_file": faiss_file,
                    "distance": distances[0][i],
                    "text": text_list[idx]
                })
                all_distances.append(distances[0][i])
                all_indices.append(idx)


    #print(f"🔎 Retrieved Texts: {retrieved_texts}")
    if not retrieved_texts:
        print("🚫 No relevant documents found.")
    return distances, indices, retrieved_texts, metadata

thread_local = threading.local()

def process_file(role, file_path, output_path, file_id, socketio):
    print(f"Processing {file_path}")
    
    if file_id not in STATUS:
        STATUS[file_id] = {"progress": 0}
    for i in range(1, 11):
        socketio.sleep(1) 
        STATUS[file_id]["progress"] = i * 10
               
        print(f"🔄 Sent progress update: {STATUS[file_id]['progress']}% for {file_id}")

        if i < 2:
            STATUS[file_id]["progress"] = 20
            file_extension = os.path.splitext(file_path)[1].lower()
            STATUS[file_id]["detailed_message"] = "📖 กำลังอ่านไฟล์..."
            # ดึงข้อมูลจากไฟล์
            if file_extension == '.txt':
                text_list = extract_text_from_txt(file_path, file_id)
            elif file_extension == '.json':
                text_list = extract_text_from_json(file_path, file_id)
            elif file_extension == '.docx':
                text_list = extract_text_from_docx(file_path, file_id)
            elif file_extension == '.pdf':
                text_list = extract_text_from_pdf(file_path, file_id)
            else:
                return None
            print(f"📜 text_list ({len(text_list)} items): {text_list[:5]}")

            text_list_split = []
            for text in text_list:
                text_list_split.extend(split_text(text))  # แยกข้อความออกเป็น chunks
                embeddings = extract_bert_embeddings(text_list_split)

            print(f"🔹 text_list_split ({len(text_list_split)} items): {text_list_split[:5]}")
            
            # ✅ ใช้ text_list เพื่อสร้าง embeddings
            """if text_list is not None:
                embeddings = extract_bert_embeddings(text_list)
                print("Before reshape, embeddings shape:", embeddings.shape)"""

        elif i < 4 :
            STATUS[file_id]["progress"] = 40
            STATUS[file_id]["detailed_message"] = "✍🏻 กำลังสร้าง Vector และ Metadata"
            # ✅ ปรับขนาดให้เป็น (N, 768)
            if len(embeddings.shape) == 3:
                embeddings = embeddings.squeeze(axis=1)

            print("After reshape, embeddings shape:", embeddings.shape)

            if embeddings.shape[1] != 768:
                raise ValueError(f"Expected embedding shape (N, 768), but got {embeddings.shape}")

            # ✅ สร้าง FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
            # ✅ สร้าง ID สำหรับแต่ละเวกเตอร์
            ids = np.arange(embeddings.shape[0]).astype(np.int64)  # ID ต้องเป็น int64

            # ✅ เพิ่มเวกเตอร์ลงใน FAISS index โดยใช้ add_with_ids
            index.add_with_ids(embeddings, ids)
            print(f"✅ Added {embeddings.shape[0]} vectors to FAISS index with IDs")
            
            # ✅ ตั้งชื่อไฟล์
            file_number = get_next_file_number(output_path)
            index_filename = os.path.join(output_path, f'file_{file_number}.faiss')
            pickle_filename = os.path.join(output_path, f'file_{file_number}.pkl')        
        elif i == 6:
            STATUS[file_id]["progress"] = 60
            STATUS[file_id]["detailed_message"] = "📂 บันทึก FAISS..."
            # ✅ บันทึก FAISS index
            try:
                faiss.write_index(index, index_filename)
            except Exception as e:
                print(f"Error saving FAISS index: {e}")
                return None
        elif i == 8:
            STATUS[file_id]["progress"] = 80
            STATUS[file_id]["detailed_message"] = "🔄 บันทึก Pickle..."
            # ✅ บันทึกข้อมูลที่เกี่ยวข้อง
            try:
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(text_list, f)
            except Exception as e:
                print(f"Error saving pickle file: {e}")
                return None
        elif i == 10:
            STATUS[file_id]["progress"] = 100 
            STATUS[file_id]["detailed_message"] = "✅ Upload complete!"
            insert_metadata(role, index_filename, "faiss", file_number, os.path.basename(file_path))

            print(f"FAISS index and pickle file saved as {index_filename} and {pickle_filename}")
       
        socketio.emit("upload_status", {"file_id": file_id, 
                                        "progress": STATUS[file_id]["progress"],
                                        "file_name": os.path.basename(file_path)})

        socketio.sleep(1)
    return text_list

STATUS = {"progress": 0, "message": "กำลังอัพโหลด"}
thread_local.STATUS = {"progress": 0, "message": "กำลังอัพโหลด"}

def upload_progress(file_id, socket_io):
    global STATUS

    # ตรวจสอบว่ามี file_id ใน STATUS หรือไม่ ถ้าไม่มีให้สร้าง
    if file_id not in STATUS:
        STATUS[file_id] = {"progress": 0, "message": "กำลังอัพโหลด..."}
    for i in range(1, 11):
        print("ค่า  I : ", i)
        time.sleep(1)
        STATUS[file_id]["progress"] = i * 10 
        if i < 2:
            STATUS[file_id]["progress"] = 20
            STATUS[file_id]["detailed_message"] = "📖 กำลังอ่านไฟล์..."
        elif i < 4:
            STATUS[file_id]["progress"] = 40
            STATUS[file_id]["detailed_message"] = "✍🏻 กำลังสร้าง Vector และ Metadata"
        elif i < 6:
            STATUS[file_id]["progress"] = 60
            STATUS[file_id]["detailed_message"] = "📂 บันทึก FAISS..."
        elif i < 8:
            STATUS[file_id]["progress"] = 80
            STATUS[file_id]["detailed_message"] = "🔄 บันทึก Pickle..."
        elif i >= 10:
            STATUS[file_id]["progress"] = 100
            STATUS[file_id]["detailed_message"] = "✅ Upload complete!"
            
        
        # Emit status to the client
        socket_io.emit('upload_status', {"file_id": file_id, **STATUS[file_id]}, broadcast=True)
   
def check_new_files(input_path, output_path, file_path, file_id):
    global read_files
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        
        if file_path not in read_files:
            process_file(file_path, output_path, file_id, socketio)
            read_files.append(file_path)

if __name__ == "__main__":
    file_id = str(uuid.uuid4())
    check_new_files()
    time.sleep(300)

"""
def loadtyphoon2(modelconfig):
    load_path = os.path.join("model_trained", modelconfig)

    typhoon2_model = AutoModelForCausalLM.from_pretrained(
        load_path,
        device_map="auto",
        local_files_only=True
    )
    typhoon2_tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
    return typhoon2_model, typhoon2_tokenizer


## Russian  Tonkey : คิดดูสิ
def typhoon2chat(query, typhoon2_model, typhoon2_tokenizer):
    messages = [
        {"role": "system", "content": "You are a male AI assistant named Tonkey created by TRU to be helpful, harmless, and honest. Tonkey is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks. Tonkey responds directly to all human messages without unnecessary affirmations or filler phrases like “Certainly!”, “Of course!”, “Absolutely!”, “Great!”, “Sure!”, etc. Specifically, Tonkey avoids starting responses with the word “Certainly” in any way. Tonkey follows this information in all languages, and always responds to the user in the language they use or request. Tonkey is now being connected with a human. Write in fluid, conversational prose, Show genuine interest in understanding requests, Express appropriate emotions and empathy. Also showing information in term that is easy to understand and visualized."},
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
    outputs = typhoon2_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    return typhoon2_tokenizer.decode(response, skip_special_tokens=True)


 
if __name__ == "__main__":
    file_id = str(uuid.uuid4())
    check_new_files()
    time.sleep(300)


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        fdpath = "documents/json_main"
        faiss_folder="vector_store"
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break      
        print("AI is generating a response...")
        distances_all, indices_all, retrieved_texts, metadata = search_faiss_in_folder(sentence, faiss_folder, top_k=3)
        print(f"Retrieved Text lastest: {retrieved_texts}")
        print(f"Metadata : {metadata}")
        #for i, text in enumerate(retrieved_texts):
        #    print(f"Item {i}: {type(text)}")
        
        context = " ".join(str(text) for text in retrieved_texts)
        print("context : ", context)                          
        input_text = f"บริบท: {context} \nคำถาม: {sentence}"

        model, tokenizer = loadtyphoon2("local_typhoon2_1b")
        response = typhoon2chat(input_text, model, tokenizer)
        print(f"answer : {response}")
"""

# เอกวิชาภาษาจีนของมหาวิทยาลัยราชภัฏเทพสตรี เรียนวิชาใดบ้างขอรายชื่อวิชาที่เรียน
# รายชื่อวิชาที่เรียนของเอกวิชาภาษาจีน มหาวิทยาลัยราชภัฏเทพสตรี
# รายชื่อกรรมการวิพากษ์หลักสูตรศิลปศาสตร์บัณฑิตสาขาวิชาภาษาจีน
# รายชื่ออาจารย์ผู้รับผิดชอบหลักสูตรศิลปศาสตร์บัณฑิตสาขาวิชาภาษาจีน
# เอกวิชาภาษาจีนของมหาวิทยาลัยราชภัฏเทพสตรี ชั้นปีที่2 เรียนอะไรบ้าง
# รายละเอียดของหลักสูตรรัฐประศาสนศาสตรบัณฑิต สาขาวิชารัฐประศาสนศาสตร์ มหาวิทยาลัยราชภัฏเทพสตรี เป็นอย่างไรครับ
# มหาวิทยาลัยราชภัฏเทพสตรี อยู่จังหวัดอะไรครับ
# อาชีพที่สามารถประกอบได้หลังสำเร็จการศึกษา ของมหาวิทยาลัยราชภัฏเทพสตรี หลักสูตรรัฐประศาสนศาสตรบัณฑิต สาขาวิชารัฐประศาสนศาสตร์ 
# คณาจารย์และบุคลากรสนับสนุนการเรียนการสอน ให้มีคุณวุฒิ คุณสมบัติ และสัดส่วนเป็นไปตามเกณฑ์ สาขาวิชารัฐประศาสนศาสตร์ มหาวิทยาลัยราชภัฏเทพสตรี

# อาชีพที่สามารถประกอบได้หลังสำเร็จการศึกษาหลักสูตรรัฐประศาสนศาสตร์บัณฑิตสาขาวิชารัฐประศาสนศาสตร์