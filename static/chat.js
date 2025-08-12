async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (!userInput.trim()) return;
    
    console.log('Sending message:', userInput); // Debug log
    
    // Display user message
    displayMessage(userInput, 'user');
    document.getElementById('user-input').value = '';
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_input: userInput
            })
        });
        
        const data = await response.json();
        console.log('Response data:', data); // Debug log
        
        if (response.ok) {
            displayMessage(data.response, 'bot');
        } else {
            displayMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Fetch error:', error); // Debug log
        displayMessage('Connection error: ' + error.message, 'error');
    }
}

function displayMessage(message, type) {
    console.log('Displaying message:', message, type); // Debug log
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) {
        console.error('Chat container not found!');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Wait for DOM to load before adding event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Enter key support
    const userInput = document.getElementById('user-input');
    if (userInput) {
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
});