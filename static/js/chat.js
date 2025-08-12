// Get references to important DOM elements
const chatMessages = document.getElementById('chatMessages');  // Container for chat messages
const chatForm = document.getElementById('chatForm');          // The chat form element
const messageInput = document.getElementById('messageInput');  // Text input for user message
const sendButton = document.getElementById('sendButton');      // Send button

// Attach event listener to the form to handle submissions (sending messages)
chatForm.addEventListener('submit', handleSubmit);

// Function to handle chat form submission
async function handleSubmit(e) {
    e.preventDefault();  // Prevent default form submit behavior (page reload)

    // Get trimmed user message from input
    const message = messageInput.value.trim();
    if (!message) return;  // Ignore empty messages

    // Display the user's message in the chat UI
    addMessage(message, 'user');

    // Clear the input field for next message
    messageInput.value = '';

    // Disable input and send button while waiting for backend response
    setLoading(true);

    try {
        // Send user message to backend via POST request
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',  // JSON content type
            },
            body: JSON.stringify({ question: message })  // Send message as JSON
        });

        // Parse JSON response from server
        const data = await response.json();

        if (response.ok) {
            // If response is successful, display bot's answer in chat
            addMessage(data.answer, 'bot');
        } else {
            // If error from server, show error message in chat
            addErrorMessage(data.error || 'An error occurred while processing your request.');
        }
    } catch (error) {
        // Handle network or other unexpected errors
        console.error('Error:', error);
        addErrorMessage('Failed to connect to the server. Please try again.');
    } finally {
        // Re-enable input and send button after processing is complete
        setLoading(false);
    }
}

// Function to add a chat message to the UI
// 'content' is the message text, 'sender' is either 'user' or 'bot'
function addMessage(content, sender) {
    const messageDiv = document.createElement('div');   // Outer message container
    messageDiv.className = `message ${sender}`;         // Add CSS classes for styling

    const messageContent = document.createElement('div');  // Inner div for text content
    messageContent.className = 'message-content';
    messageContent.textContent = content;

    const timeDiv = document.createElement('div');      // Timestamp div
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();  // Current local time

    // Assemble the message div structure
    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(timeDiv);

    // Append message to chat messages container
    chatMessages.appendChild(messageDiv);

    // Auto-scroll chat to the latest message
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to add an error message to chat UI
function addErrorMessage(error) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';  // CSS class for error styling
    errorDiv.textContent = error;
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;  // Scroll to bottom
}

// Function to add a "loading" indicator message (shows when waiting for bot reply)
function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot';  // Styled as bot message
    loadingDiv.id = 'loadingMessage';      // Assign an ID for easy removal

    const loadingContent = document.createElement('div');
    loadingContent.className = 'message-content loading';

    // HTML showing "Thinking..." with animated dots
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

// Function to remove the loading indicator from chat UI
function removeLoadingMessage() {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// Helper function to toggle the loading state:
// disables/enables input and send button, and adds/removes loading indicator
function setLoading(loading) {
    messageInput.disabled = loading;
    sendButton.disabled = loading;

    if (loading) {
        addLoadingMessage();
    } else {
        removeLoadingMessage();
    }
}

// Focus the message input when the page loads
messageInput.focus();

// Allow sending message by pressing Enter key (without Shift)
messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();            // Prevent new line
        chatForm.dispatchEvent(new Event('submit'));  // Trigger form submission
    }
});