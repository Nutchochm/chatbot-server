/* TRAINING PANEL */
document.addEventListener("DOMContentLoaded", function () {
    const techniqueSelect = document.getElementById("fine-tune-technique");
    const trainingSettings_1 = document.getElementById("trainingSettings_1");
    const trainingSettings_2 = document.getElementById("trainingSettings_2");

    // ฟังก์ชันซ่อน/แสดงกล่องตั้งค่า
    function updateTrainingSettings() {
        if (techniqueSelect.value === "Lora") {
            trainingSettings_1.style.display = "block";
            trainingSettings_2.style.display = "none";
        } else if (techniqueSelect.value === "Bert") {
            trainingSettings_1.style.display = "none";
            trainingSettings_2.style.display = "block";
        } else {
            trainingSettings_1.style.display = "none";
            trainingSettings_2.style.display = "none";
        }
    }

    // เรียกใช้งานเมื่อลงหน้า HTML
    updateTrainingSettings();

    // ติดตามการเปลี่ยนค่า select
    techniqueSelect.addEventListener("change", updateTrainingSettings);
});


document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("uploadForm").addEventListener("submit", async function (event) {
        event.preventDefault();

        const uploadStatus = document.getElementById("uploadStatus");
        uploadStatus.innerHTML = "Uploading...";

        const formData = new FormData();
        const dataset = document.getElementById("dataset").files[0];

        if (!dataset) {
            uploadStatus.innerHTML = "Please select a file to upload.";
            return;
        }

        document.getElementById("submittrain").disabled = true;

        formData.append("dataset", dataset);

        try {
            const response = await fetch("/upload_dataset", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                uploadStatus.innerHTML = `<span style="color: green;">${result.message}</span>`;
                document.getElementById("submittrain").disabled = false;
                document.getElementById("submittrain").dataset.filename = result.filename;
            } else {
                uploadStatus.innerHTML = `<span style="color: red;">${result.message}</span>`;
            }
        } catch (error) {
            uploadStatus.innerHTML = `<span style="color: red;">An error occurred: ${error.message}</span>`;
        }
    });
});
function toggleMode(mode) {
    const clientModeBtn = document.getElementById("clientModeBtn");
    const serverModeBtn = document.getElementById("serverModeBtn");

    const clientModeSection = document.getElementById("clientModeSection");
    const serverModeSection = document.getElementById("serverModeSection");

    const inputSelector = document.getElementById("inputSelector");
    const serverModelList = document.getElementById("serverModelList");

    if (mode === "client") {
        console.log("Switching to Client Mode");
        clientModeBtn.classList.add("active");
        serverModeBtn.classList.remove("active");

        clientModeSection.style.display = "block";
        serverModeSection.style.display = "none";

        inputSelector.disabled = false;
        serverModelList.disabled = true;
        fileMode.disabled = false;
        folderMode.disabled = false;
    } else if (mode === "server") {
        console.log("Switching to Server Mode");
        serverModeBtn.classList.add("active");
        clientModeBtn.classList.remove("active");

        clientModeSection.style.display = "none";
        serverModeSection.style.display = "block";

        inputSelector.disabled = true;
        serverModelList.disabled = false;
        fileMode.disabled = true;
        folderMode.disabled = true;

        // Fetch models from the server
        fetch("/get_models")
            .then(response => response.json())
            .then(data => {
                serverModelList.innerHTML = ""; // Clear previous options
                let defaultOption = document.createElement("option");
                defaultOption.value = "";
                defaultOption.textContent = "-- Choose a Model --";
                serverModelList.appendChild(defaultOption);

                if (data.models) {
                    data.models.forEach(model => {
                        let option = document.createElement("option");
                        option.value = model;
                        option.textContent = model;
                        serverModelList.appendChild(option);
                    });
                } else {
                    console.error("Error:", data.error);
                }
            })
            .catch(error => console.error("Error fetching models:", error));
    }
}


// Add event listeners for the mode buttons
document.getElementById("clientModeBtn").addEventListener("click", () => toggleMode("client"));
document.getElementById("serverModeBtn").addEventListener("click", () => toggleMode("server"));

document.addEventListener("DOMContentLoaded", function () {
    const inputSelector = document.getElementById("inputSelector");
    const fileMode = document.getElementById("fileMode");
    const folderMode = document.getElementById("folderMode");

    // Toggle between file and folder modes
    fileMode.addEventListener("change", function () {
        inputSelector.removeAttribute("webkitdirectory");
        inputSelector.setAttribute("multiple", "true");
    });

    folderMode.addEventListener("change", function () {
        inputSelector.setAttribute("webkitdirectory", "");
        inputSelector.removeAttribute("multiple");
    });

    document.getElementById("loadmodelForm").addEventListener("submit", function (event) {
        event.preventDefault(); 
        const uploadmodelStatus = document.getElementById("uploadmodelStatus");
        uploadmodelStatus.innerHTML = "Uploading...";

        const formData = new FormData();

        const clientModeSection = document.getElementById("clientModeSection");
        const serverModeSection = document.getElementById("serverModeSection");

        if (clientModeSection.style.display !== "none") {
            // Client Mode - Use inputSelector
            const inputSelector = document.getElementById("inputSelector");

            if (inputSelector.files.length > 0) {
                const modelFolderName = inputSelector.files[0].webkitRelativePath.split("/")[0];
                formData.append("model_folder_name", modelFolderName);
            } else {
                uploadmodelStatus.innerHTML = "No model selected!";
                return;
            }

        } else if (serverModeSection.style.display !== "none") {
            // Server Mode - Use serverModelList
            const serverModelList = document.getElementById("serverModelList");

            if (serverModelList.value) {
                formData.append("model_folder_name", serverModelList.value);
            } else {
                uploadmodelStatus.innerHTML = "No model selected!";
                return;
            }
        }

        // Append any additional files if needed
        const inputSelector = document.getElementById("inputSelector");
        if (inputSelector.files.length > 0) {
            for (let i = 0; i < inputSelector.files.length; i++) {
                formData.append("model", inputSelector.files[i]);
            }
        }

        // Send files to the backend
        fetch("/upload_model", {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    uploadmodelStatus.innerHTML = `<p style="color: green;">Upload successful! Files are processed.</p>`;
                } else {
                    uploadmodelStatus.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
                }
            })
            .catch(err => console.error("Error uploading files:", err));
    });
});

// Handle training form submission
document.getElementById("submittrain").addEventListener("click", function (event) {
    event.preventDefault();

    const selectedTechnique = document.getElementById('technique').value;
    const modelFolderName = document.getElementById("inputSelector").files[0]?.webkitRelativePath.split("/")[0];
    const datasetFilename = this.dataset.filename;
    const modelName = document.getElementById("model_name").value.trim();
    const modelPath = document.getElementById("output_path").value.trim();
    const device = document.getElementById("device").value.trim();

    if (!validateTrainingSettings()) {
        alert("Please fill in all the fields.");
        return;
    }
    if (!modelFolderName) {
        alert("Model folder name could not be determined.");
        return;
    }

    const formData = new FormData(document.getElementById("trainForm"));
    formData.append("model_folder_name", modelFolderName);
    formData.append("dataset_path", datasetFilename);
    formData.append("model_name", modelName);
    formData.append("output_path", modelPath);

    // Add technique-specific parameters
    if (selectedTechnique === "Fine-Tuning Lora") {
        formData.append('lora_r', '16');
        formData.append('lora_alpha', '32');
        formData.append('lora_dropout', '0.1');
        formData.append('learning_rate', '1e-4');
        formData.append('batch_size', '4');
        formData.append('epochs', '3');
        formData.append('device', device);
    
    
        // Send training request
        fetch("/finetune_lora", {
            method: "POST",
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {       
                    const uploadedModelFolder = document.getElementById("uploadedModelFolder");
                    if (uploadedModelFolder) {
                        uploadedModelFolder.value = data.model_folder_name;
                    } else {
                        console.warn("Element with id 'uploadedModelFolder' not found.");
                    }
                }    
                document.getElementById("loramessages").textContent = "Training started successfully.";
                console.log("Response:", data);
            })
            .catch(error => {
                console.error("Error:", error);
                alert("An error occurred during training.");
            });
    } 
    
    if (selectedTechnique === "Fine-Tuning Bert") {
        formData.append('bert_lr', '2e-5');
        formData.append('bert_tr_bs', '8');
        formData.append('bert_ev_bs', '8');
        formData.append('bert_epochs', '10');
        formData.append('bert_decay', '0.01');

        // Send Bert-specific training request
        fetch("/finetune_bert", {
            method: "POST",
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const uploadedModelFolder = document.getElementById("uploadedModelFolder");
                    if (uploadedModelFolder) {
                        uploadedModelFolder.value = data.model_folder_name;
                    } else {
                        console.warn("Element with id 'uploadedModelFolder' not found.");
                    }
                }
                document.getElementById("bertmessages").textContent = "Bert training started successfully.";
                console.log("Response:", data);
            })
            .catch(error => {
                console.error("Error:", error);
                alert("An error occurred during Bert training.");
            });
        }
    
});

function validateTrainingSettings() {
    const requiredFields = [
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "output_path",
        "model_name",
    ];

    return requiredFields.every(id => document.getElementById(id).value.trim() !== "");
}