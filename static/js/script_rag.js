/*
document.getElementById("document").addEventListener("change", function () {
    let fileInput = document.getElementById("document");
    let fileNameDisplay = document.querySelector(".file-name");
    let file = fileInput.files[0];

    if (!file) {
        fileNameDisplay.textContent = "ยังไม่ได้เลือกไฟล์";
        return;
    }

    const allowedTypes = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "application/json"];

    if (!allowedTypes.includes(file.type)) {
        alert("ไฟล์ที่เลือกต้องเป็น PDF, DOCX, TXT หรือ JSON เท่านั้น!");
        fileInput.value = "";
        fileNameDisplay.textContent = "ยังไม่ได้เลือกไฟล์";
        return;
    }

    fileNameDisplay.textContent = file.name;
});*/
//var socket = io.connect('http://' + document.domain + ':' + location.port);
var socket = io.connect('http://127.0.0.1:5000');

socket.on("upload_status", function(data) {
    console.log(`Received progress update for file ${data.file_id}: ${data.progress}%`);
    console.log(`Received from backend : ${data}`);
 
    // สมมุติว่าเรามี progress bar ใน HTML
    const progressBar = document.getElementById('progress-bar' + data.file_id);
    if (progressBar) {
        progressBar.style.width = data.progress + '%'; // อัปเดตความกว้างของ progress bar
    }

    // อัปเดตข้อความแสดง progress
    const progressText = document.getElementById('process-file-name' + data.file_id);
    if (progressText) {
        progressText.innerText = `Processing... ${data.progress}%`;
    }
});

socket.on("upload_complete", function(data) {
    console.log(`File ${data.file_id} completed`);
    const progressText = document.getElementById('progress-text-' + data.file_id);
    if (progressText) {
        progressText.innerText = `Completed!`;
    }
});

document.getElementById("progress-container").style.display = "none";

document.getElementById("document").addEventListener("change", function(event) {
    let fileList = event.target.files;
    let fileNames = Array.from(fileList).map(file => `"${file.name}"`).join("\n");
    document.querySelector(".file-name").innerText = fileList.length ? fileNames : "ยังไม่ได้เลือกไฟล์";
    document.getElementById("progress-bar").style.width = "0%";
    document.getElementById("progress-bar").innerText = "0%";
});


document.getElementById("documentsubmit").addEventListener("click", function(event) {
    event.preventDefault();
    
    let formData = new FormData();
    let files = document.getElementById("document").files;
    let foldername = document.getElementById("folderid").value;

    if (files.length > 0) {
        for (let i = 0; i < files.length; i++) {
            formData.append("files[]", files[i]);
        }
        formData.append("foldername", foldername);

        let progressBar = document.getElementById("progress-bar");
        let uploadProgress = document.getElementById("progress-container");
        uploadProgress.style.display = "block";

        fetch("/upload_rag", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => { throw new Error(text) });
            }
            return response.json();
        })
        .then(data => {
            //checkUploadStatus(data.file_id);
            showFlashMessage(data.status, data.message);
            if (data.status === "success") {                
                fetchFlashMessages();
            } else {
                showFlashMessage("error", data.message);
            }
        })
        .catch(error => {
            console.error("Error:", error);
            showFlashMessage("error", "เกิดข้อผิดพลาดในการอัปโหลด");
        });

        // ฟังก์ชันสำหรับการอัปเดต progress
        socket.on('upload_status', function(data) {
            console.log("📡 Received upload status:", data);
            if (data && data.file_id) {
                // ค้นหา progress ของไฟล์
                if (data.progress) {
                    let progress = data.progress;
                    let message = data.detailed_message || "กำลังอัปโหลด...";

                    requestAnimationFrame(() => {
                        progressBar.style.width = progress + "%";  // อัปเดตความกว้าง
                        progressBar.textContent = `${progress}%`; // อัปเดตข้อความ
                    });

                    console.log(`✅ Progress updated: ${progress}%`);
                    console.log(`✅ message updated: ${message}%`);
                }
            }
        });
    } else {
        showFlashMessage("error", "กรุณาเลือกไฟล์");
    }
});


function showFlashMessage(category, message) {
    const flashContainer = document.getElementById("flash-container");
    const alertDiv = document.createElement("div");
    alertDiv.className = `alert alert-${category}`;
    alertDiv.innerText = message;
    flashContainer.appendChild(alertDiv);

    setTimeout(() => alertDiv.remove(), 5000);
}

function fetchFlashMessages() {
    fetch("/get_flash_messages")
    .then(response => response.json())
    .then(data => {
        const flashContainer = document.getElementById("flash-container");
        flashContainer.innerHTML = "";
        data.messages.forEach(([category, message]) => {
            const alertDiv = document.createElement("div");
            alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            flashContainer.appendChild(alertDiv);

            setTimeout(() => {
                alertDiv.classList.remove("show");
                alertDiv.classList.add("fade");
                setTimeout(() => alertDiv.remove(), 500);
            }, 3000);        
        });
    })
    .catch(error => console.error("Fetch error:", error));
}

function updateProgress(progress, progress_message) {
    const progressBar = document.getElementById("progress-bar");
    const progressContainer = document.getElementById("progress-container");
    const processFileName = document.getElementById("process-file-name");

    progressBar.style.width = progress + "%";
    progressBar.textContent = progress + "%";

    processFileName.innerText = progress_message || "กำลังอัพโหลด...";
    console.log("Update Progress:", progress, progress_message);
    progressContainer.style.display = progress < 100 ? "block" : "none";
}
/*
function checkUploadStatus(file_id) {
    let progressBar = document.getElementById("progress-bar");
    let progressContainer = document.getElementById("progress-container");
    let fileNameLabel = document.getElementById("process-file-name");

    // ✅ เชื่อม WebSocket เพื่ออัพเดทสถานะแบบเรียลไทม์
    socket.on("upload_status", function (data) {
        console.log("📡 Received upload status:", data);

        if (data.file_id === file_id) {
            if (data.file_name) {
                fileNameLabel.innerText = `📂 กำลังประมวลผลไฟล์: ${data.file_name}`;
            }

            progressBar.style.width = data.progress + "%";
            progressBar.innerText = data.progress + "%";

            if (data.progress === 10) {
                progressContainer.style.display = "block";
                fileNameLabel.style.display = "block";
            }

            if (data.progress >= 100) {
                console.log("🎯 Process completed, hiding progress bar.");
                setTimeout(() => {
                    progressContainer.style.display = "none";
                    fileNameLabel.innerText = "✅ เสร็จสิ้น";
                }, 1500);
            }
        }
    });
}*/
