document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll("textarea").forEach(textarea => {
        adjustHeight(textarea);
        textarea.addEventListener("input", function () {
            adjustHeight(this);
        });
    });

    const collectionDropdown = document.getElementById("collection_dropdown");

    let savedCollection = localStorage.getItem("selectedCollection");
    if (savedCollection) {
        collectionDropdown.value = savedCollection;
    }

    window.saveSelectedCollection = function (value) {
        localStorage.setItem("selectedCollection", value);
    };

    collectionDropdown.addEventListener("change", function () {
        fetch("/update_session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ collection_selected: this.value })
        }).then(() => fetchCollections());
    });

    let noDataRow = document.getElementById("no-data-row");
    // ปุ่มเพิ่มแถวใหม่
    const addRowButton = document.getElementById("addRowButton");
    const table = document.getElementById("qna_table").getElementsByTagName('tbody')[0];
    
    addRowButton.addEventListener("click", function() {
        if (noDataRow){
            noDataRow.remove();
        }        

        const row = table.insertRow();
        row.innerHTML = `
            <td><input type="text" class="idInput" readonly></td>
            <td><textarea class="questionInput" oninput="adjustHeight(this)" placeholder="โปรดใส่คำถาม"></textarea></td>
            <td><textarea class="userquestionInput" oninput="adjustHeight(this)" placeholder="โปรดใส่อินพุต"></textarea></td>
            <td><textarea class="corransInput" oninput="adjustHeight(this)" placeholder="โปรดใส่คำตอบที่ถูกต้อง"></textarea></td>
            <td><textarea class="refInput" oninput="adjustHeight(this)" placeholder="โปรดใส่แหล่งที่มา"></textarea></td>
            <td>
                <i class="fa fa-edit editButton"></i>
                <i class="fa fa-save saveButton"></i>
                <i class="fa fa-trash deleteButton"></i>
            </td>
        `;
    });

    // ตรวจจับการคลิกปุ่มบนตาราง
    table.addEventListener("click", function(e) {
        const row = e.target.closest("tr");
                
        // ฟังก์ชันบันทึกข้อมูล (Insert หรือ Update)
        if (e.target && e.target.matches(".saveButton")) {
            const idInput = row.querySelector(".idInput");
            const question = row.querySelector(".questionInput").value;
            const userquestion = row.querySelector(".userquestionInput").value;
            const correctAnswer = row.querySelector(".corransInput").value;
            const reference = row.querySelector(".refInput").value;            

            const isEdit = idInput.value !== "";

            fetch(isEdit ? "/edit_row" : "/add_row", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    collection_selected: collectionDropdown.value,
                    _id: isEdit ? Number(idInput.value) : undefined,
                    question: question,
                    input: userquestion,
                    correct_answer: correctAnswer,
                    reference: reference
                })
            })
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    if (!isEdit && data.id) {
                        idInput.value = data.id;                        
                    }
                    if (data.status === "success") {
                        fetchFlashMessages();
                    }
                })
                .catch(error => console.error(error));
        }

        // ฟังก์ชันลบข้อมูล
        if (e.target && e.target.matches(".deleteButton")) {
            const id = row.querySelector(".idInput").value;

            if (!id) {
                console.error("No ID found for deletion");
                return;
            }

            fetch("/delete_row", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    collection_selected: collectionDropdown.value,
                    _id: id
                })
            })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => Promise.reject(err));
                    }
                    return response.json();
                })
                .then(data => {
                    console.log(data);
                    if (data.status === "success") {
                        row.remove();
                        fetchFlashMessages();
                    }
                })
                .catch(error => console.error("Fetch error:", error));
        }
    });
});

function adjustHeight(textarea){
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + ".px";
}

function sortTable(columnIndex, iconElement) {
    let table = document.getElementById("qna_table");
    let tbody = table.getElementsByTagName("tbody")[0];
    let rows = Array.from(tbody.getElementsByTagName("tr"));
    let isAscending = iconElement.getAttribute("data-order") !== "desc"; // ถ้าไม่ใช่ desc ให้เรียงน้อยไปมาก
    iconElement.setAttribute("data-order", isAscending ? "desc" : "asc");

    function isNumeric(value) {
        return !isNaN(value) && !isNaN(parseFloat(value));
    }

    rows.sort((rowA, rowB) => {
        let cellA = rowA.getElementsByTagName("td")[columnIndex].querySelector("textarea, input").value.trim();
        let cellB = rowB.getElementsByTagName("td")[columnIndex].querySelector("textarea, input").value.trim();

        let isNumberA = isNumeric(cellA);
        let isNumberB = isNumeric(cellB);

        if (isNumberA && isNumberB) {
            return isAscending ? parseFloat(cellA) - parseFloat(cellB) : parseFloat(cellB) - parseFloat(cellA);
        }

        if (isNumberA) return isAscending ? -1 : 1;
        if (isNumberB) return isAscending ? 1 : -1;

        return isAscending ? cellA.localeCompare(cellB, 'th') : cellB.localeCompare(cellA, 'th');
    });

    tbody.innerHTML = "";
    rows.forEach(row => tbody.appendChild(row));

    document.querySelectorAll(".btn_sorttable").forEach(icon => {
        icon.classList.remove("fa-sort-up", "fa-sort-down");
        icon.classList.add("fa-sort");
    });

    if (iconElement) {
        iconElement.classList.remove("fa-sort");
        iconElement.classList.add(isAscending ? "fa-sort-up" : "fa-sort-down");
    } else {
        console.error("iconElement is undefined! Check if <i> tag is properly passed.");
    }
}


function sendRequest(url) {
    fetch(url, { method: "POST" })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            fetchFlashMessages();
        } else {
            console.error("Error:", data);
        }
    })
    .catch(error => console.error("Request failed:", error));
}

function fetchFlashMessages() {
    fetch("/get_flash_messages")
    .then(response => {
        if (!response.ok) throw new Error(`HTTP Error! Status: ${response.status}`);
        return response.json();
    })
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
