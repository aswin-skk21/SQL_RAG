// Get DOM elements
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');

// Add event listener for form submission
chatForm.addEventListener('submit', handleSubmit);

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    messageInput.value = '';
    
    // Disable input and button while processing
    setLoading(true);
    
    try {
        // Send message to backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: message })
        });

        const data = await response.json();
        
        if (response.ok) {
            // Add bot response to chat
            addMessage(data.answer, 'bot');
        } else {
            // Handle error
            addErrorMessage(data.error || 'An error occurred while processing your request.');
        }
    } catch (error) {
        console.error('Error:', error);
        addErrorMessage('Failed to connect to the server. Please try again.');
    } finally {
        // Re-enable input and button
        setLoading(false);
    }
}

// Add a message to the chat
function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    
    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(timeDiv);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add an error message
function addErrorMessage(error) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = error;
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add loading indicator
function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot';
    loadingDiv.id = 'loadingMessage';
    
    const loadingContent = document.createElement('div');
    loadingContent.className = 'message-content loading';
    loadingContent.innerHTML = `
        Thinking
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    loadingDiv.appendChild(loadingContent);
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove loading indicator
function removeLoadingMessage() {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// Set loading state
function setLoading(loading) {
    messageInput.disabled = loading;
    sendButton.disabled = loading;
    
    if (loading) {
        addLoadingMessage();
    } else {
        removeLoadingMessage();
    }
}

// Focus on input when page loads
messageInput.focus();

// Handle Enter key
messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});
