document.addEventListener('DOMContentLoaded', function () {
    const radioButtons = document.querySelectorAll('input[type="radio"]');
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    //const answerRadios = document.querySelectorAll('input[id^="answer_radio_"]');
    const max_num_radio = window.max_num_radio;  

    // Update the value for radio_button_chkans dynamically
    const answerRadios = document.querySelectorAll('input[name="radio_button_chkans"]');
    answerRadios.forEach(function (radio, index) {
        // Assign value starting from maxNumRadio + 1
        radio.value = max_num_radio + index + 1;
        });

    console.log('Updated answer_radio values:');
    answerRadios.forEach(radio => console.log(radio.id, radio.value));
    

    let checkedCheckboxes = 0;
    let guestFolder = '';
    let studentFolder = '';
    let adminFolder = '';
    let inspectFolder = '';
    let exclusiveFolder = '';
    let chkansFolder = '';

    // Initially disable all radio buttons
    radioButtons.forEach(function (radio) {
        radio.disabled = true;
    });

    // Enable/disable radio buttons based on checkbox state
    checkboxes.forEach(function (checkbox) {
        checkbox.addEventListener('change', function () {
            const folderName = this.value;
            const relatedRadios = document.querySelectorAll(`input[name="radio_button_${folderName}"]`);
            const relatedAnswerModelButton = document.getElementById(`answer_radio_${folderName}`);

            if (this.checked) {
                if (checkedCheckboxes >= max_num_radio) {
                    this.checked = false;
                    alert(`You can only select up to ${max_num_radio} folders.`);
                    return;
                }
                checkedCheckboxes++;

                if (relatedAnswerModelButton) {
                    relatedAnswerModelButton.disabled = false;
                }
            } else {
                checkedCheckboxes--;

                if (relatedAnswerModelButton) {
                    relatedAnswerModelButton.disabled = true;
                    relatedAnswerModelButton.checked = false;
                }
    
                const relatedRadios = document.querySelectorAll(`input[name="radio_button_${folderName}"]`);
                relatedRadios.forEach(radio => {
                    radio.disabled = true;
                    radio.checked = false;
                });
            }

            relatedRadios.forEach(function (radio) {
                const sameValueSelected = Array.from(radioButtons).some(function (otherRadio) {
                    return otherRadio !== radio && otherRadio.value === radio.value && otherRadio.checked;
                });

                if (checkbox.checked && !sameValueSelected) {
                    radio.disabled = false;
                } else {
                    radio.disabled = true;
                    radio.checked = false;
                }
            });
        });
    });

    // Logic to allow only one radio button to be selected at a time from checked folders
    radioButtons.forEach(function (radio) {
        radio.addEventListener('change', function () {
            if (this.checked) {
                const folderName = this.name.replace("radio_button_", "");
                const answerfolderName = this.name.replace("answer_radio_", "");
                const selectedValue = this.value;
                console.log("Folder Name:", folderName);
                console.log("Selected Value:", selectedValue);

                if (selectedValue === '0') {
                    guestFolder = folderName; 
                } else if (selectedValue === '1') {
                    studentFolder = folderName; 
                } else if (selectedValue === '2') {
                    adminFolder = folderName; 
                } else if (selectedValue === '3') {
                    inspectFolder = folderName;
                } else if (selectedValue === '4') {
                    exclusiveFolder = folderName;                
                } else {
                    chkansFolder = answerfolderName;
                }

                console.log('Updated guestFolder:', guestFolder);
                console.log('Updated studentFolder:', studentFolder);
                console.log('Updated adminFolder:', adminFolder);
                console.log('Updated inspectFolder:', inspectFolder);
                console.log('Updated exclusiveFolder:', exclusiveFolder);
                console.log('Updated chkansFolder:', chkansFolder);
    
                // Construct the config object with different radioIDs and folders
                let config = {
                    guest: {
                        role: "guest",
                        folder: guestFolder || folderName,
                        radioID: 0
                    },
                    student: {
                        role: "student",
                        folder: studentFolder || folderName, 
                        radioID: 1
                    },
                    admin: {
                        role: "admin",
                        folder: adminFolder || folderName,  
                        radioID: 2
                    },
                    inspector: {
                        role: "inspector",
                        folder: inspectFolder || folderName,
                        radioID: 3
                    },
                    exclusive: {
                        role: "exclusive",
                        folder: exclusiveFolder || folderName,
                        radioID: 4
                    },
                    chkans: {
                        role: "admin",
                        folder: chkansFolder,
                        activate: 1
                    }
                };
    
                // Debugging the constructed config
                console.log('Config to send:', config);

                fetch('/submit_adminconfig', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'  
                    },
                    body: JSON.stringify(config)
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Data submitted successfully:', data);                    
                })
                .catch(error => {
                    console.error('Error submitting data:', error);
                });

                radioButtons.forEach(function (otherRadio) {
                    if (otherRadio !== radio && otherRadio.value === selectedValue) {
                        otherRadio.disabled = true;
                        otherRadio.checked = false;
                    }
                });
            } else {
                const selectedValue = this.value;

                radioButtons.forEach(function (otherRadio) {
                    if (otherRadio.value === selectedValue) {
                        otherRadio.disabled = false;
                    }
                });
            }
        });
    });

    
    // Answer radio logic (single selection across folders)
    answerRadios.forEach(answerRadio => {
        answerRadio.addEventListener('change', function () {
            if (this.checked) {
                    const answerfolderName = this.name.replace("answer_radio_", "");
                    chkansFolder = answerfolderName;
                    console.log(`Answer model enabled for folder: ${answerfolderName}`);
                }
        });
    });
    
    const submitButton = document.getElementById('submitButton');
    submitButton.addEventListener('click', function () {        
        if (!guestFolder && !studentFolder && !adminFolder && !inspectFolder && !exclusiveFolder && !chkansFolder) {
            alert('Please select at least one permission.');
            return;
        }
    
        let config = {
            guest: {
                role: "guest",
                folder: guestFolder,
                radioID: 0
            },
            student: {
                role: "student",
                folder: studentFolder,
                radioID: 1
            },
            admin: {
                role: "admin",
                folder: adminFolder,
                radioID: 2
            },
            inspector: {
                role: "inspector",
                folder: inspectFolder,
                radioID: 3
            },
            exclusive: {
                role: "exclusive",
                folder: exclusiveFolder,
                radioID: 4
            },
            chkans: {
                role: "admin",
                folder: chkansFolder,
                activate: 1
            }
        };
    
        console.log('Final Config on Submit:', config);
    
        fetch('/submit_adminconfig', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'  
            },
            body: JSON.stringify(config)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Data submitted successfully:', data);
            if (data.message) {
                alert(data.message);
            }
            window.location.href = data.redirect;                
        })
        .catch(error => {
            console.error('Error submitting data:', error);
            alert(error.message || "An error occurred while submitting the data.");
        });
    });

    // Reset button functionality
    const resetButton = document.getElementById('resetButton');
    resetButton.addEventListener('click', function () {
        checkboxes.forEach(function (checkbox) {
            checkbox.checked = false;
        });

        radioButtons.forEach(function (radio) {
            radio.checked = false;
            radio.disabled = true;
        });

        checkedCheckboxes = 0; 
    });    
});
