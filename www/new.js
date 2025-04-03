$(document).ready(function () {

    // Display messages from Python directly in the chat
    eel.expose(displayMessageInChat);
    function displayMessageInChat(message) {
        var chatBox = document.getElementById("chat-canvas-body");
        if (message.trim() !== "") {
            chatBox.innerHTML += `
                <div class="row justify-content-start mb-4">
                    <div class="width-size">
                        <div class="receiver_message">${message}</div>
                    </div>
                </div>`;
            // Scroll to the bottom of the chat box
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    }

});