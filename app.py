from flask import Flask, get_flashed_messages, render_template, redirect, url_for, request, jsonify, session, flash, Response
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from chat import gen_response, load_model, loadtyphoon2, typhoon2chat
from chkans_app import load_chkans_model, tokenize_input, get_embedding, compute_similarity, rating_scores
from fewshot_prompt import construct_prompt, cal_perplexity, cal_bleu_rouge
from rag_transformers import search_faiss_in_folder, preprocess_query, process_file
from app_training_technique import train_finetune_lora, excel_csv, finetune_bert, prepare_few_shot_rag, train_few_shot_rag, extract_text_from_pdf, create_json_from_pdf, load_pdf_as_dataset
from pymongo import MongoClient, ReturnDocument
from datetime import datetime
from pytz import timezone
import numpy as np
import threading
import uuid
import re
import json
import os

app = Flask(__name__, static_folder="static")
app.secret_key = "TRU@administrator"
socketio = SocketIO(app, cors_allowed_origins="*")

# MongoDB Connection
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["chatbot_db"]
    messages_collection = db["messages"]
    topic_collection = db["topics"]
    users_collection = db["users"]    
    authen_collection = db["authen"]
    guest_collection = db["qa_guest"]
    student_collection = db["qa_student"]
    admin_collection = db["qa_admin"]
    inspector_collection = db["qa_inspector"]
    exclusive_collection = db["qa_exclusive"]
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")

# Timezone
bkk_tz = timezone("Asia/Bangkok")  

# Define default configuration
default_config = {
    "guest": {
        "role": "guest",
        "folder": "",
        "radioID": ""
    },
    "student": {
        "role": "student",
        "folder": "",
        "radioID": ""
    },
    "admin": {
        "role": "admin",
        "folder": "",
        "radioID": ""
    },
    "inspector": {
        "role": "inspector",
        "folder": "",
        "radioID": ""
    },
    "exclusive": {
        "role": "exclusive",
        "folder": "",
        "radioID": ""
    },
    "chkans": {
        "role": "admin",
        "folder": "",
        "activate": ""
    },
}


@app.route("/chat")
def chat():
    #topics = topic_collection.distinct("topic_name")
    topics=""
    return render_template("chat.html", topic=topics)

@app.route("/")
@app.route("/home", methods=["GET", "POST"])
def home():
    login()

    if "user" not in session:
        return redirect(url_for("login"), code=301)
    
    return render_template("home.html")

@app.route('/activate_model', methods=['POST'])
def initialize():
    expected_token = load_token()
    data = request.get_json()
    configmodel = load_adminconfig(CONFIG_FILE)

    # Extract token from received data
    received_token = data.get("token")
    auth_value = data["role"]
    resp_acc = data["response_access"]
    user_query = data.get("answer")        
    user_id = data.get("user_id")
    user_name = data.get("user_name")
    ### auth_value 0 is guest and fewshot
    ### auth_value 1 is student and rag
    ### auth_value 2 is admin and rag
    ### auth_value 3 is inspector and rag
    ### auth_value 4 is exclusive and rag

    adminconfig_model_name, examples = check_role(auth_value, configmodel)
    print("check_role model name: ", adminconfig_model_name, "check_role examples : ", examples)

    if received_token != expected_token: 
        return jsonify({"error": "Invalid token."}), 401
    else:
        faiss_folder = ""
        typhoon2_model, typhoon2_tokenizer = loadtyphoon2(adminconfig_model_name)
                
        data = {
            "answer": user_query,
            "role": auth_value,
            "token": received_token, 
            "user_id": user_id,
            "user_name": user_name
        }

        topic_name = re.split(r'[.!?]', user_query)[0].strip()
        if not topic_name:
            topic_name = user_query.split()[0]
            data["topic_name"] = topic_name
        
        if resp_acc == 1:
            # FEW-SHOT
            generated_prompt = construct_prompt(user_query, examples)            
            response = typhoon2chat(generated_prompt, typhoon2_model, typhoon2_tokenizer)
        elif resp_acc == 2:
            # RAG
            if adminconfig_model_name == "student":
                faiss_folder = os.path.join(VECTOR_FOLDER, adminconfig_model_name)
                print(f"faiss_folder: {faiss_folder}")
            elif adminconfig_model_name == "admin":
                faiss_folder = os.path.join(VECTOR_FOLDER, adminconfig_model_name)
                print(f"faiss_folder: {faiss_folder}")
            if adminconfig_model_name == "inspector":
                faiss_folder = os.path.join(VECTOR_FOLDER, adminconfig_model_name)
                print(f"faiss_folder: {faiss_folder}")
            if adminconfig_model_name == "exclusive":
                faiss_folder = os.path.join(VECTOR_FOLDER, adminconfig_model_name)
                print(f"faiss_folder: {faiss_folder}")

            preprocess_text = preprocess_query(user_query)
            print(f"Pre process text: {preprocess_text}")
            distances, indices, retrieved_texts, metadata = search_faiss_in_folder(preprocess_text, faiss_folder)
            print(f"🎯 ค่าที่คืนมา: indices = {indices}, distances = {distances}")
            print(f"🔎 Retrieved Texts: {retrieved_texts}")
            print(f"🔎 Metadata: {metadata}")

            prompts = f"""{retrieved_texts} Q: {user_query} A: """
            print(f"rag + prompt : {prompts}")
            response = typhoon2chat(prompts, typhoon2_model, typhoon2_tokenizer)
        else:            
            response = typhoon2chat(user_query, typhoon2_model, typhoon2_tokenizer)

        message_id, topic_id = get_id(user_id, user_name, topic_name)

        insert_to_users(user_id, user_name, topic_id)
        insert_to_message(message_id, user_id, topic_id, topic_name, user_query, response)

        perplexity_scored = cal_perplexity(response, typhoon2_model, typhoon2_tokenizer)
        #bleu, rouge = cal_bleu_rouge(Correct_answer, response)

        print(f"Model Response : {response}")
        print(f"Perplexity Scores : {perplexity_scored}")
        #print(f"BLEU Score : {bleu}")
        #print(f"ROUGE Score: {rouge}")           
        
        return jsonify({
            "response": response
        }), 200
    
def check_role(auth_value, configmodel):
    try:
        if auth_value == 0:
            adminconfig_model_name = configmodel.get("guest", {}).get("folder")
            collection = guest_collection
        elif auth_value == 1:
            adminconfig_model_name = configmodel.get("student", {}).get("folder")
            collection = student_collection
        elif auth_value == 2:
            adminconfig_model_name = configmodel.get("admin", {}).get("folder")
            collection = admin_collection
        elif auth_value == 3:
            adminconfig_model_name = configmodel.get("inspector", {}).get("folder")
            collection = inspector_collection
        elif auth_value == 4:
            adminconfig_model_name = configmodel.get("Exclusive", {}).get("folder")
            collection = exclusive_collection
        else:
            return None, None
        
        examples = [(re.sub(r'\d+$', '', doc["question"]).strip() if doc["question"] else '', 
                     re.sub(r'\d+$', '', doc["correct_answer"]).strip()) if doc["correct_answer"] else ''
                    for doc in collection.find({}, {"question": 1, "correct_answer": 1, "_id": 0})]

        print(f"Model Name: {adminconfig_model_name}, Total Examples: {len(examples)}")
        
        return adminconfig_model_name, examples
    except Exception as e:
        print(f"Error: {e}")

def load_token():
    with open("appconfig.json", "r") as file:
        data = json.load(file)
        return data["token"]

"""def load_adminconfig(adminfile):
    try:
        # Check if file exists and can be read
        if not os.path.exists(adminfile):
            print(f"File not found: {adminfile}")
            with open(adminfile, 'w') as file:
                json.dump(default_config, file, indent=4)
            return default_config        
        with open(adminfile, 'r') as file:
            config = json.load(file)            
            
            if 'student' not in config:
                config['student'] = {"role": "", "folder": "", "radioID": 0}
            if 'admin' not in config:
                config['admin'] = {"role": "", "folder": "", "radioID": 0}
            if 'inspector' not in config:
                config['inspector'] = {"role": "", "folder": "", "radioID": 0}
            if 'exclusive' not in config:
                config['exclusive'] = {"role": "", "folder": "", "radioID": 0}
            if 'chkans' not in config:
                config['chkans'] = {"role": "admin", "folder": "", "activate": 0}
            
            if 'student' in config['student']:
                config['student']['radioID'] = int(config['student'].get('radioID', 0))
            if 'admin' in config['admin']:
                config['admin']['radioID'] = int(config['admin'].get('radioID', 0))
            if 'inspector' in config['inspector']:
                config['inspector']['radioID'] = int(config['inspector'].get('radioID', 0))
            if 'exclusive' in config['exclusive']:
                config['exclusive']['radioID'] = int(config['exclusive'].get('radioID', 0))
            if 'chkans' in config['chkans']:
                config['chkans']['activate'] = int(config['chkans'].get('activate', 0))

            with open(adminfile, 'w') as file:
                json.dump(config, file, indent=4)
            return config        
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config"""
# Route to handle login
@app.route("/login", methods=["GET", "POST"])
def login():
    # If user is already logged in, redirect them to the dashboard
    if 'user' in session:
        return redirect(url_for('fewshotdb'), code=301)
    
    if request.method == 'POST':
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)
        # Find the user in the database
        user = authen_collection.find_one({"username": username})
        is_valid = check_password_hash(hashed_password, password)
        print("checked the password:", is_valid)

        # Verify the password
        if user and is_valid:
            session['user'] = username
            session['password'] = password
            flash("Login successful!", "success")
            return redirect(url_for('home'), code=301)
        else:
            flash("Invalid username or password", "danger")
            return render_template('login.html') 

    return render_template('login.html') 

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"), code=301) 

@app.route('/dashboard')
def dashboard(collection_db):
    if 'user' not in session:
        return redirect(url_for('fewshotdb'), code=301)

    # Fetch entries from MongoDB to display on dashboard
    entries = collection_db.find().sort('_id', -1)
    return render_template('dashboard.html', entries=entries)

def load_adminconfig(adminfile):
    try:
        # Check if file exists and can be read
        if not os.path.exists(adminfile):
            print(f"File not found: {adminfile}")
            with open(adminfile, 'w') as file:
                json.dump(default_config, file, indent=4)
            return default_config
        
        with open(adminfile, 'r') as file:
            config = json.load(file)
        
        # Initialize missing sections in the config with default values
        for role in ['guest', 'student', 'admin', 'inspector', 'exclusive', 'chkans']:
            if role not in config:
                config[role] = default_config[role]
        
        # Save updated config back to the file
        with open(adminfile, 'w') as file:
            json.dump(config, file, indent=4)
        
        return config

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {adminfile}: {e}")
        return default_config
    
def insert_to_users(user_id, user_name, topic_id):
    message = {
        "user_id": user_id,
        "user_name": user_name,
        "topic_id": topic_id,
        "created_at": datetime.now(bkk_tz)
    }
    users_collection.insert_one(message)

def insert_to_message(message_id, user_id, topic_id, topic_name, user_message, ai_message):
    message = {
        "message_id": message_id,
        "user_id": user_id,
        "topic_id": topic_id,
        "topic_name": topic_name,
        "user_message_id": f"user_{message_id}",
        "user_message": user_message,
        "ai_message_id": f"ai_{message_id}",
        "ai_message": ai_message,
        "timestamp": datetime.now(bkk_tz)
    }
    messages_collection.insert_one(message)
    

def insert_to_topic(user_id, user_name, topic_name):
    # Get the current month
    current_month = datetime.now(bkk_tz).strftime("%Y-%m")
    
    # Check the latest topic for the user
    last_topic = topic_collection.find_one(
        {"user_id": user_id, "user_name": user_name, "topic_name": topic_name},
        sort=[("created_at", -1)]
    )
    
    # Reset topic_id if the month has changed or no previous topic exists
    if not last_topic or last_topic["created_at"].strftime("%Y-%m") != current_month:
        new_topic_id = "0001"
    else:
        new_topic_id = str(int(last_topic["topic_id"]) + 1).zfill(4)
    
    # Insert the new topic
    topic_collection.insert_one({
        "user_id": user_id,
        "user_name": user_name,
        "topic_id": new_topic_id,
        "topic_name": topic_name,
        "message_seq": 1,
        "created_at": datetime.now(bkk_tz)
    })
    return new_topic_id

def get_id(user_id, user_name, topic_name):
    # Find an existing topic for the user and topic name
    existing_topic = topic_collection.find_one(
        {"user_id": user_id, "user_name": user_name, "topic_name": topic_name},
        sort=[("created_at", -1)]
    )
    
    # If no topic exists or the month has changed, insert a new topic
    if not existing_topic or existing_topic["created_at"].strftime("%Y-%m") != datetime.now(bkk_tz).strftime("%Y-%m"):
        topic_id = insert_to_topic(user_id, user_name, topic_name)
    else:
        topic_id = existing_topic["topic_id"]
    
    # Increment the message sequence for the topic
    topic = topic_collection.find_one_and_update(
        {"topic_name": topic_name, "user_name": user_name, "topic_id": topic_id},
        {"$inc": {"message_seq": 1}},
        return_document=True
    )
    if not topic:
        raise ValueError(f"Failed to update sequence for topic: {topic_name}")
    
    return str(topic["message_seq"]).zfill(4), topic_id


## ไม่ get ค่า ใน mongodb
@app.route('/chatbot_db/topics', methods=['GET'])
def get_topics():
    user_id = request.args.get('user_id')
    message_id = request.args.get('message_id')
    topic_id = request.args.get('topic_id')
    topic_name = request.args.get('topic_name')

    # Find topics related to the user (assuming topics have user_id or user_name)
    topics = list(topic_collection.find({"user_id": user_id, 
                                         "message_id": message_id,
                                         "topic_id": topic_id,
                                         "topic_name": topic_name
                                         }))
    topic_names = [topic['topic_name'] for topic in topics]
    
    return jsonify({"topics": topic_names}), 200

# Route to fetch messages for a topic
@app.route('/chatbot_db/messages/<topic_name>', methods=['GET'])
def get_messages(topic_name):
    user_id = request.args.get('user_id')

    # Fetch messages for the given topic and user_id
    messages = list(messages_collection.find({"topic_name": topic_name, "user_id": user_id}))
    
    # Sort by 'user_id' and 'ai_message' (adjust sort fields as per your schema)
    messages.sort(key=lambda msg: (msg.get("user_id"), msg.get("ai_message")))
    
    # Convert _id to string for JSON serialization
    for message in messages:
        message["_id"] = str(message["_id"])

    return jsonify({"topic_name": topic_name, "messages": messages}), 200

#########################################################################################################
################################                Home Page                ################################
#########################################################################################################

@app.route("/train", methods=["GET", "POST"])
def training():
    if "user" not in session:
        return redirect(url_for("login"), code=301)
    try:        
        max_num_radio = 4
        selected_radio_buttons = {}
        # Check if MODEL_FOLDER exists and is a directory
        if os.path.exists(MODEL_TRAIN_FOLDER) and os.path.isdir(MODEL_TRAIN_FOLDER):
            folder_names = [f for f in os.listdir(MODEL_TRAIN_FOLDER) if os.path.isdir(os.path.join(MODEL_TRAIN_FOLDER, f))]
            #print("Folder Found.", folder_names)
        else:
            folder_names = ["Directory not found."]

    except FileNotFoundError:
        folder_names = ["Directory not found."]
    except Exception as e:
        folder_names = [f"Error: {e}"]

    admin_config_data  = load_adminconfig(CONFIG_FILE)
    auth_value = 1
    adminconfig_model_name, examples = check_role(auth_value, admin_config_data)
    
    folder_radio_map = {folder: list(range(0, max_num_radio)) for folder in folder_names}

    #entries = list(examples.find({}))
    #entries_list = [entry for entry in entries if isinstance(entry.get('_id'), int)]
    #print("Entries from DB:", entries_list)
    
    collections = db.list_collection_names()
    qa_collections = [col for col in collections if col.startswith('qa_')]
    
    # ตรวจสอบค่าที่เลือกจาก dropdown
    collection_name = request.args.get("collection")
   
    return render_template("train.html")

#########################################################################################################
################################              training Page              ################################
#########################################################################################################

UPLOAD_DATASET = 'upload_dataset'
UPLOAD_MODEL = 'upload_model'
MODEL_TRAINED = 'model_trained'

# 1. Upload dataset
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    try:
        if not os.path.exists(UPLOAD_DATASET):
            os.makedirs(UPLOAD_DATASET, exist_ok=True)
        
        dataset = request.files.get("dataset")
        if not dataset:
            return jsonify({"success": False, "message": "No dataset file provided."})

        # Save file locally
        dataset_path = os.path.join(UPLOAD_DATASET, dataset.filename)
        dataset.save(dataset_path)
        print(f"Dataset uploaded to: {dataset_path}")

        return jsonify({"success": True, "message": "Dataset uploaded successfully.", "dataset_path": dataset_path, "filename": dataset.filename})

    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})
        

# 2. Upload model from a directory
@app.route("/upload_model", methods=["POST"])
def upload_model():
    try:
        # Ensure upload directory exists
        if not os.path.exists(UPLOAD_MODEL):
            os.makedirs(UPLOAD_MODEL)
        
        # Get the model folder name from the form
        model_folder_name = request.form.get("model_folder_name")
        if not model_folder_name:
            return jsonify({"success": False, "message": "Model folder name not provided."})
        
        # Create the model-specific folder if it doesn't exist
        model_folder_path = os.path.join(UPLOAD_MODEL, model_folder_name)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # Retrieve the expected file names from the frontend
        file_names = request.form.get("file_names")
        if not file_names:
            return jsonify({"success": False, "message": "No filenames provided."})

        expected_files = json.loads(file_names)

        # Retrieve the uploaded files from the request
        files = request.files.getlist("model_files[]")
        if not files:
            return jsonify({"success": False, "message": "No files provided."})

        # List the uploaded filenames
        uploaded_files = [os.path.basename(file.filename) for file in files]

        # Validate if the uploaded files match the expected filenames
        if set(uploaded_files) != set(expected_files):
            return jsonify({"success": False, "message": "Filename mismatch."})

        # Save files into the model folder
        saved_files = []
        for file in files:
            try:
                if file and file.filename:                                      
                    # Define the path where the file will be saved
                    file_path = os.path.join(model_folder_path, os.path.basename(file.filename))
                    file.save(file_path)
                    saved_files.append(file.filename)
                    print(f"Model File is uploaded to: {file_path}")
            except Exception as e:
                print(f"Error saving file {file.filename}: {str(e)}")

        return jsonify({"success": True, "message": "Files uploaded successfully.", "files": saved_files, "model_folder_path": model_folder_path})

    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})
    
# 3. Load models from a directory
@app.route("/get_models", methods=["GET"])
def get_models():
    try:
        if not os.path.exists(MODEL_TRAINED):
            return jsonify({"error": "Folder not found"}), 404

        # Get only directories (folders) inside MODEL_FOLDER
        models = [d for d in os.listdir(MODEL_TRAINED) if os.path.isdir(os.path.join(MODEL_TRAINED, d))]
        
        return jsonify({"models": models})  # Return folder names as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/finetune_lora', methods=['POST', 'GET'])
def finetune_lora():
    try:
        dataset_name = request.form.get('dataset_path')
        newmodel_path = request.form.get('output_path')
        newmodel_name = request.form.get('model_name') 
        
        print(f"Received dataset_path: {dataset_name}")
        print(f"Received output_path: {newmodel_path}")
        print(f"Received model_name: {newmodel_name}")

        dataset_path = os.path.join(UPLOAD_DATASET, dataset_name)

        if not newmodel_name or not dataset_path or not newmodel_path:
            return jsonify({"success": False, "message": "Missing required fields!"})

        if not os.path.exists(dataset_path):
            return jsonify({"success": False, "message": f"Dataset path does not exist: {dataset_path}"})

        if not os.path.exists(newmodel_path):
            os.makedirs(newmodel_path, exist_ok=True)

        csvform = excel_csv(dataset_path)

        save_model_dir = os.path.join(newmodel_path, newmodel_name)
        os.makedirs(save_model_dir, exist_ok=True)

        # Log the save path
        print(f"Model will be saved at: {save_model_dir}")

        lora_r = request.form.get('lora_r')
        lora_alpha = request.form.get('lora_alpha')
        lora_dropout = request.form.get('lora_dropout')
        learning_rate = request.form.get('learning_rate')
        batch_size = request.form.get('batch_size')
        epochs = request.form.get('epochs')
        receive_device = request.form.get('device')
        print(f"first {receive_device}")

        modelupload_folder_name = request.form.get("model_folder_name")
        print("Form keys received:", request.form.keys())
        print(f"Model upload folder name: {modelupload_folder_name}")
        
        # Create the model-specific folder if it doesn't exist
        model_folder_path = os.path.join(UPLOAD_MODEL, modelupload_folder_name)
        if not os.path.exists(model_folder_path):
            return jsonify({"success": False, "message": "Please upload base model"})
        
        
        # Call the finetune_lora function with the parameters
        train_finetune_lora(
            model_path=model_folder_path,
            dataset_path=csvform,
            save_path=save_model_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            device=receive_device
        )       
    #return redirect(url_for("/training", model_name=newmodel_name))
    
        return jsonify({"success": True, "model_folder_name": save_model_dir})
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})


app.route("/finetune_bert", methods=["POST", 'GET'])
def finetune_bert():
    try:
        dataset_name = request.form.get('dataset_path')
        newmodel_path = request.form.get('output_path')
        newmodel_name = request.form.get('model_name') 
        
        print(f"Received dataset_path: {dataset_path}")
        print(f"Received output_path: {newmodel_path}")
        print(f"Received model_name: {newmodel_name}")

        dataset_path = os.path.join(UPLOAD_DATASET, dataset_name)

        if not newmodel_name or not dataset_path or not newmodel_path:
            return jsonify({"success": False, "message": "Missing required fields!"})

        if not os.path.exists(dataset_path):
            return jsonify({"success": False, "message": f"Dataset path does not exist: {dataset_path}"})

        if not os.path.exists(newmodel_path):
            os.makedirs(newmodel_path, exist_ok=True)

        csvform = excel_csv(dataset_path)

        save_model_dir = os.path.join(newmodel_path, newmodel_name)
        os.makedirs(save_model_dir, exist_ok=True)

        # Log the save path
        print(f"Model will be saved at: {save_model_dir}")

        bert_lr = int(request.form.get('bert_lr', 2e-5))
        bert_tr_batchsize = int(request.form.get('bert_tr_batchsize', 8))
        bert_ev_batchsize = float(request.form.get('bert_ev_batchsize',8))
        train_epoch = int(request.form.get('train_epoch', 10))
        w_decay = float(request.form.get('w_decay', 0.01))
        receive_device = request.form.get('device')
        print(f"first {receive_device}")

        modelupload_folder_name = request.form.get("model_folder_name")
        print(f"model upload name : {modelupload_folder_name}")
        
        # Create the model-specific folder if it doesn't exist
        model_folder_path = os.path.join(UPLOAD_MODEL, modelupload_folder_name)
        if not os.path.exists(model_folder_path):
            return jsonify({"success": False, "message": "Please upload base model"})

        finetune_bert(model_name=model_folder_path, 
                  save_path=save_model_dir, 
                  csv_path=csvform,
                  bert_lr=bert_lr, 
                  bert_tr_batchsize=bert_tr_batchsize, 
                  bert_ev_batchsize=bert_ev_batchsize, 
                  train_epoch=train_epoch, 
                  w_decay=w_decay,
                  device=receive_device
                  )
        
        return jsonify({"success": True, "message": f"Training completed! Model saved to: {save_model_dir}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})
        

############################################################################################
##########################              admin config              ##########################
############################################################################################

CONFIG_FILE = "configmodel.json"
MODEL_TRAIN_FOLDER = os.path.join(os.getcwd(), 'model_trained')
print(f"MODEL_FOLDER path: {MODEL_TRAIN_FOLDER}")

@app.route("/adminconfig", methods=["GET"])
def admin_config():
    try:        
        max_num_radio = 5
        selected_radio_buttons = {}
        # Check if MODEL_FOLDER exists and is a directory
        if os.path.exists(MODEL_TRAIN_FOLDER) and os.path.isdir(MODEL_TRAIN_FOLDER):
            folder_names = [f for f in os.listdir(MODEL_TRAIN_FOLDER) if os.path.isdir(os.path.join(MODEL_TRAIN_FOLDER, f))]
            #print("Folder Found.", folder_names)
        else:
            folder_names = ["Directory not found."]
    except FileNotFoundError:
        folder_names = ["Directory not found."]
    except Exception as e:
        folder_names = [f"Error: {e}"]

    admin_config_data  = load_adminconfig(CONFIG_FILE)
    print("Before update:", admin_config_data)

    # Mapping folder names to available radio button values (1 to max_num_radio)
    folder_radio_map = {folder: list(range(0, max_num_radio)) for folder in folder_names}
    print("Radio Map : ", folder_radio_map)

    return render_template(
        "adminconfig.html",
        admin_config_form=True, 
        folders=folder_names,
        max_num_radio=max_num_radio,
        folder_radio_map=folder_radio_map,
        selected_radio_buttons=selected_radio_buttons,
        admin_config=admin_config_data
    )

@app.route('/submit_adminconfig', methods=['POST'])
def submit_adminconfig():
    try:
        if request.content_type != 'application/json':
            return jsonify({"error": "Invalid Content-Type, must be application/json"}), 415

        data = request.get_json()
        print("Received data:", data) 

        admin_config = load_adminconfig(CONFIG_FILE)
        print("Loaded admin config before update:", admin_config)

        admin_config.update(data)

        with open(CONFIG_FILE, 'w') as config_file:
            json.dump(admin_config, config_file, indent=5)

        return jsonify({"message": "Config saved successfully", "redirect": "/adminconfig"}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process the request"}), 400


########################################################################
##############                check_answer                ##############
########################################################################

@app.route('/checker', methods=['POST'])
def check_answer():
    data = request.get_json()

    input_question = data.get("input_question")
    correct_answer = data.get("correct_answer")
    input_answer = data.get("input_answer")
    received_token = data.get("token")

    if not input_question or not correct_answer or not input_answer:
        return jsonify({"error": "Missing required fields."}), 400
    
    expected_token = load_token()
    
    if received_token != expected_token: 
        return jsonify({"error": "Invalid token."}), 401
    
    ans_models = get_ans_models(CONFIG_FILE)
    print("check answer model named : ", ans_models)  
        
    if not ans_models:
        return jsonify({"error": "No valid model folder found in configuration."}), 500
    
    model_name = os.path.join(MODEL_TRAIN_FOLDER, ans_models)
    if not os.path.exists(model_name):
        print(f"Model not found: {model_name}")

    ca_model, ca_tokenizer = load_chkans_model(model_name)

    input_encodings = tokenize_input(input_question, input_answer, ca_tokenizer)
    correct_encodings = tokenize_input(input_question, correct_answer, ca_tokenizer)
    
    input_embedding = get_embedding(input_encodings, ca_model)
    correct_embedding = get_embedding(correct_encodings, ca_model)

    print(correct_answer)
    print(input_answer)
    
    similarity_score = compute_similarity(input_embedding, correct_embedding)
    print(f"Similarity Score: {similarity_score:.2f}")

    ratings = rating_scores(similarity_score)

    # Threshold for human-acceptable similarity
    if similarity_score >= 0.8:        
        print(f"Answer is correct or highly similar. given score: {ratings}")
    elif 0.5 <= similarity_score < 0.8:
        print("Answer is partially correct.", ratings)
    else:
        print("Answer is incorrect.", ratings)

    return jsonify({"Similarity Score": similarity_score, "Ratings": ratings})
    

def get_ans_models(config_file):
    if not os.path.exists(config_file):
        return {"error": f"Config file {config_file} does not exist."}

    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            print(f"Loaded config: {config}")
            
            if "chkans" not in config or not isinstance(config["chkans"], dict):
                return {"error": "'chkans' is missing or not a list."}
            
            chkans = config["chkans"]
            ans_model = chkans.get("folder")
            
            if not ans_model:
                return {"error": "'folder' key is missing in 'chkans'."}
            
            return ans_model

    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


########################################################################
##############               fewshot prompt               ##############
########################################################################

@app.route("/fewshotdb", methods=["GET"]) 
def fewshotdb(): 
    collections = db.list_collection_names()
    qa_collections = [col for col in collections if col.startswith('qa_')]
    collection_name = request.args.get('collection_name', session.get("collection_selected", "default"))
    
    print("fewshotdb session name:", collection_name)
    
    entries = list(db[collection_name].find().sort("_id", 1)) if collection_name != "default" else []
    for entry in entries:
        entry['_id'] = str(entry['_id'])
    
    return render_template("fewshotdb.html", collections=qa_collections, collection_name=collection_name, entries=entries)


def get_next_id(collection_name):
    last_entry = collection_name.find_one(
        sort=[('_id', -1)]
    )
    
    last_id = last_entry['_id'] if last_entry else 0

    counter = db['counter'].find_one_and_update(
        {'_id': 'fewshot_counter'},
        {'$set': {'counter': last_id + 1}},
        upsert=True,
        return_document=True
    )

    return counter['counter']

@app.route('/clear_session', methods=['POST'])
def clear_session():
    print("ลบ session เก่าละนะจ๊ะ")
    session.clear()
    return redirect(url_for('get_collections'))

@app.route('/get_collections', methods=['GET'])
def get_collections():
    clear_session()
    collection_name = request.args.get('collection_name', "default")
    
    collections = db.list_collection_names()
    qa_collections = [col for col in collections if col.startswith('qa_')]

    print("collection name of get_collections : ", collection_name) 

    if collection_name == "default" or collection_name not in qa_collections:
        flash("ไม่พบ Collection หรือไม่มีข้อมูล", "warning")
        return redirect(url_for('fewshotdb'))
    
    get_entries = list(db[collection_name].find().sort("_id", 1))
    for entry in get_entries:
        entry['_id'] = str(entry['_id'])

    session['get_entries'] = get_entries

    return redirect(url_for('fewshotdb', collections=qa_collections, collection_name=collection_name, entries=get_entries))

@app.route('/update_session', methods=['POST'])
def update_session():
    req_data = request.json
    collection_selected = req_data.get("collection_selected")
    data_store["session"] = collection_selected
    return jsonify({"message": "Session updated", "selected": collection_selected})

data_store = {}

@app.route('/add_row', methods=['POST'])
def add_row():
    data = request.json
    collection_selected = data.get("collection_selected")

    if not collection_selected:
        return jsonify({"message": "Missing collection_selected"}), 400

    collection_name = db[collection_selected]

    required_keys = ["question", "input", "correct_answer", "reference"]
    if not all(key in data for key in required_keys):
        return jsonify({"message": "Missing required data"}), 400

    try:
        new_id = get_next_id(collection_name)
        existing_entry = collection_name.find_one({'_id': new_id})
    
        if existing_entry:
            print(f"⚠️ Duplicate _id detected: {new_id}")
            return jsonify({'error': f'Duplicate ID {new_id}, try again'}), 400    

        result = collection_name.insert_one({
            '_id': new_id,
            'question': data['question'],
            'input': data['input'],
            'correct_answer': data['correct_answer'],
            'reference': data['reference']
        })
        flash("เพิ่มข้อมูลสำเร็จ!", "success")
        # ส่ง ID ใหม่กลับไปที่ Frontend
        return jsonify({"status": "success", "id": new_id})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500    

@app.route('/edit_row', methods=['POST'])
def edit_row():
    try:
        data = request.json
        collection_selected = data.get("collection_selected")
        collection_name = db[collection_selected]

        _id = data.get('_id')
        if _id is None:
            return jsonify({'error': 'No ID provided'}), 400

        # แปลง _id เป็น int
        try:
            _id = int(_id)
        except ValueError:
            return jsonify({'error': 'Invalid ID format. It must be an integer.'}), 400

        # กำหนดค่าที่ต้องการอัปเดต
        update_fields = {
            "question": data.get("question"),
            "input": data.get("input"),
            "correct_answer": data.get("correct_answer"),
            "reference": data.get("reference")
        }

        # ทำการอัปเดตข้อมูลใน MongoDB
        result = collection_name.update_one({"_id": _id}, {"$set": update_fields})
        
        if result.modified_count > 0:
            flash(f"แก้ไขข้อมูล {_id} สำเร็จ!", "info")
            return jsonify({"status": "success", "id": _id})
        else:
            return jsonify({'error': 'No changes made or ID not found'}), 400

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/delete_row', methods=['POST'])
def delete_row():
    try:
        data = request.json
        collection_selected = data.get("collection_selected")
        collection_name = db[collection_selected]

        if not collection_selected:
            return jsonify({'error': 'Collection not specified'}), 400

        _id = data.get('_id')

        if _id is None:
            return jsonify({'error': 'No ID provided'}), 400
        
        try:
            _id = int(_id)  
        except ValueError:
            return jsonify({'error': 'Invalid ID format. It must be an integer.'}), 400

        result = collection_name.delete_one({'_id': _id})        

        if result.deleted_count > 0:
            flash(f"ลบข้อมูล {_id} สำเร็จ!", "danger")
            return jsonify({"status": "success", "id": _id})
        else:
            return jsonify({'error': 'ID not found'}), 400

    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500
    
@app.route("/get_flash_messages", methods=["GET"])
def get_flash_messages():
    messages = get_flashed_messages(with_categories=True)
    return jsonify(messages=messages)
    
def get_entry(q, a):
    collection_name = session.get("collection_selected", "default")
    entry = collection_name.find_one({
        "question": q,
        "correct_answer": a
    })

    if entry:
        return f"Found Entry: {entry}"
    else:
        return "No matching entry found."

STATUS = {"progress": 0, "message": "กำลังอัพโหลด"}
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'json'}
UPLOAD_RAG_FOLDER = "documents/contents/"
VECTOR_FOLDER = os.path.join("documents", "vectors")
UPLOAD_STATUS = {}
"""
def upload_progress(file_id, socketio):
    global STATUS

    # ตรวจสอบว่ามี file_id ใน STATUS หรือไม่ ถ้าไม่มีให้สร้าง
    if file_id not in STATUS:
        STATUS[file_id] = {"progress": 0, "message": "Uploading..."}
    for i in range(1, 11):
        socketio.sleep(1)
        STATUS[file_id]["progress"] = i * 10 
        if i < 2:
            STATUS[file_id]["detailed_message"] = "📖 กำลังอ่านไฟล์..."
        elif i < 4:
            STATUS[file_id]["detailed_message"] = "✍🏻 กำลังสร้าง Vector และ Metadata"
        elif i == 6:
            STATUS[file_id]["detailed_message"] = "📂 บันทึก FAISS..."
        elif i == 8:
            STATUS[file_id]["detailed_message"] = "🔄 บันทึก Pickle..."
        elif i == 10:
            STATUS[file_id]["detailed_message"] = "✅ Upload complete!"
        
        # Emit status to the client
        emit('upload_status', {"file_id": file_id, **STATUS[file_id]}, broadcast=True)
"""

@app.route("/rag")
def rag():
    folders = [f for f in os.listdir(VECTOR_FOLDER) if os.path.isdir(os.path.join(VECTOR_FOLDER, f))]
    return render_template("rag.html", folders=folders)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_rag", methods=['POST'])
def upload_rag():
    global STATUS    
    try:
        if 'files[]' not in request.files:
            print("No files in the request", request.files)
            return jsonify({"status": "error", "message": "No file part"})
        
        files = request.files.getlist('files[]')    
        fold_role = request.form.get("foldername", "default")
        if not files:
            return jsonify({"status": "error", "message": "No selected file"})

        documents_folder = os.path.join(UPLOAD_RAG_FOLDER, fold_role)
        vector_fold_role = os.path.join(VECTOR_FOLDER, fold_role)
        os.makedirs(documents_folder, exist_ok=True)
        os.makedirs(vector_fold_role, exist_ok=True)
        
        for index, file in enumerate(files, start=1):
            if file.filename == '':
                continue
            if allowed_file(file.filename):
                file_path = os.path.join(documents_folder, file.filename)
                file.save(file_path)            

                # เรียกใช้ฟังก์ชัน process_file โดยตรงโดยไม่ใช้ threading
                file_id = str(uuid.uuid4())
                UPLOAD_STATUS[file_id] = {"progress": 0}
                process_file(fold_role, file_path, vector_fold_role, file_id, socketio)
            else:
                return jsonify({"status": "error", "message": f"Invalid file type: {file.filename}"}), 400

        return jsonify({"status": "success", "message": "Files uploaded successfully"})
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})
    
    
def send_upload_status(file_id, progress, detailed_message=""):
    print(f"📡 Sending progress: {file_id} -> {progress}% {detailed_message}")
    socketio.emit("upload_status", {"file_id": file_id, "progress": progress, "detailed_message": detailed_message})

    

if __name__ == "__main__":
    #app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
