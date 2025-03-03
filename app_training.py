from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import json
import torch
from app_training_technique import finetune_lora, excel_csv, csv_to_huggingface_dataset

app = Flask(__name__, static_folder='static')

@app.get("/")
def index_get():
    return render_template("training.html")

UPLOAD_DATASET = 'upload_dataset'
UPLOAD_MODEL = 'upload_model'

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
        

# 2. Upload model or load models from a directory
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


@app.route('/train_lora', methods=['POST', 'GET'])
def train_lora():
    try:
        dataset_path = request.form.get('dataset_path')
        newmodel_path = request.form.get('output_path')
        newmodel_name = request.form.get('model_name') 
        
        print(f"Received dataset_path: {dataset_path}")
        print(f"Received output_path: {newmodel_path}")
        print(f"Received model_name: {newmodel_name}")


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
        
        lora_r = int(request.form.get('lora_r', 16))
        lora_alpha = int(request.form.get('lora_alpha', 32))
        lora_dropout = float(request.form.get('lora_dropout', 0.1))
        learning_rate = float(request.form.get('learning_rate', 1e-4))
        batch_size = int(request.form.get('batch_size', 4))
        epochs = int(request.form.get('epochs', 3))
        receive_device = request.form.get('device')
        print(f"first {receive_device}")

        # Create the model-specific folder if it doesn't exist
        model_folder_path = os.path.join(UPLOAD_MODEL, 'local_typhoon2_1b')
        if not os.path.exists(model_folder_path):
            return jsonify({"success": False, "message": "Please upload base model"})
        

        # Call the finetune_lora function with the parameters
        save_path = finetune_lora(
            model_name_or_path=model_folder_path,
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
        
        return jsonify({"success": True, "message": f"Training completed! Model saved to: {save_path}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})

# main driver function
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
