class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            loadingDiv: document.getElementById("loading"),
            topicDropdown: document.getElementById('topicDropdown') 
        };

        this.state = false;
        this.message = [];
        this.user_id = '';
        this.user_name = '';
    }

    // Initialize the chatbox
    display() {
        const { openButton, chatBox, sendButton, topicDropdown } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const inputField = chatBox.querySelector('input');
        inputField.addEventListener("keyup", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                this.onSendButton(chatBox);
            }
        });

        topicDropdown.addEventListener('change', (event) => {
            const selectedTopic = event.target.value;
            this.handleTopicChange(selectedTopic); 
        });

        this.loadTopics();
    }

    toggleState(chatBox) {
        this.state = !this.state;
        chatBox.classList.toggle('chatbox--active', this.state);
    }

    // Fetch topics based on user details (user_id, user_name)
    loadTopics() {
        const userIdField = document.querySelector('#user_id');
        const userNameField = document.querySelector('#user_name');

        if (userIdField && userNameField) {
            this.user_id = userIdField.value.trim();
            this.user_name = userNameField.value.trim();
        }

        // Fetch topics from MongoDB based on user_id and user_name
        fetch(`/get_topics?user_id=${this.user_id}&user_name=${this.user_name}`)
            .then(response => response.json())
            .then(data => {
                const topicDropdown = this.args.topicDropdown;
                topicDropdown.innerHTML = ''; // Clear previous options

                if (data.topics && data.topics.length > 0) {
                    data.topics.forEach(topic => {
                        const option = document.createElement('option');
                        option.value = topic.name;
                        option.textContent = topic.name;
                        topicDropdown.appendChild(option);
                    });
                } else {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'No topics available';
                    topicDropdown.appendChild(option);
                }
            })
            .catch(error => {
                console.error('Error fetching topics:', error);
            });
    }

    // Handle the topic change (fetch messages for selected topic)
    handleTopicChange(topic_name) {
        if (topic_name) {
            fetch(`/get_messages?user_id=${this.user_id}&topic_name=${topic_name}`)
                .then(response => response.json())
                .then(data => {
                    const messageDisplay = document.getElementById('messageDisplay');
                    messageDisplay.innerHTML = '';

                    const messages = data.messages;

                    if (messages.length === 0) {
                        messageDisplay.innerHTML = 'No messages available.';
                    } else {
                        messages.forEach(message => {
                            const messageElement = document.createElement('div');
                            messageElement.textContent = message.text;
                            messageDisplay.appendChild(messageElement);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error fetching messages:', error);
                });
        }
    }

    // Handle the send button
    onSendButton(chatBox) {
        const textField = chatBox.querySelector('input');
        const text1 = textField.value.trim();
        if (!text1) return;

        const msg1 = { name: this.user_name, message: text1 };
        this.message.push(msg1);

        const loadingMessage = { name: 'TRU_assistant', message: "." };
        this.message.push(loadingMessage);
        this.updateChatText(chatBox);

        let dotCount = 0;
        const loadingInterval = setInterval(() => {
            dotCount = (dotCount % 3) + 1;
            loadingMessage.message = ".".repeat(dotCount);
            this.updateChatText(chatBox);
        }, 500);

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then((response) => response.json())
            .then((data) => {
                clearInterval(loadingInterval);
                this.message.pop();
                const msg2 = { name: 'TRU_assistant', message: data.answer };
                this.message.push(msg2);
                this.updateChatText(chatBox);
                textField.value = "";
                textField.style.height = "40px";
            })
            .catch((error) => {
                clearInterval(loadingInterval);
                console.error('Error:', error);
                this.message.pop();
                const msg2 = { name: 'TRU_assistant', message: "Something went wrong. Please try again!" };
                this.message.push(msg2);
                this.updateChatText(chatBox);
            });

        // Save the message to the MongoDB database
        const messages_collection = { user_id: this.user_id, user_name: this.user_name, message: text1 };

        fetch('/chatbot_db/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ messages_collection })
        })
            .then(response => response.json())
            .then(data => {
                console.log(`Message saved in room: ${data.room_name}`);
                console.log(`Response: ${data.message}`);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }

    // Update the chat with new messages
    updateChatText(chatBox) {
        const chatMessages = chatBox.querySelector('.chatbox__messages');
        chatMessages.innerHTML = this.message
            .slice()
            .reverse()
            .map((item) =>
                item.name === "TRU_assistant"
                    ? `<div class="messages__item messages__item--visitor">${item.message}</div>`
                    : `<div class="messages__item messages__item--operator">${item.message}</div>`
            )
            .join('');

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

const chatbox = new Chatbox();
chatbox.display();
