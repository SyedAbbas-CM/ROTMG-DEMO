/**
 * ChatSystem.js - Client-side chat interface for sending commands and messages
 */

import { MessageType } from '../shared/messages.js';

export class ChatSystem {
    constructor(networkManager) {
        this.networkManager = networkManager;
        this.isVisible = false;
        this.chatHistory = [];
        this.maxHistory = 50;
        
        // Create UI elements
        this.createChatUI();
        this.setupEventHandlers();
        
        // Register for incoming chat messages
        if (networkManager) {
            networkManager.on(MessageType.CHAT_MESSAGE, (data) => {
                this.displayMessage(data.sender || 'Unknown', data.message || '', data.color || '#FFFFFF');
            });
        }
        
        console.log('[ChatSystem] Initialized');
    }
    
    createChatUI() {
        // Create chat container
        this.chatContainer = document.createElement('div');
        this.chatContainer.id = 'chatContainer';
        this.chatContainer.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 400px;
            max-height: 300px;
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #444;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: white;
            z-index: 1000;
            display: none;
        `;
        
        // Create chat messages area
        this.chatMessages = document.createElement('div');
        this.chatMessages.id = 'chatMessages';
        this.chatMessages.style.cssText = `
            padding: 10px;
            max-height: 250px;
            overflow-y: auto;
            word-wrap: break-word;
        `;
        
        // Create chat input
        this.chatInput = document.createElement('input');
        this.chatInput.id = 'chatInput';
        this.chatInput.type = 'text';
        this.chatInput.placeholder = 'Type a message or command (press Enter to send, Esc to close)...';
        this.chatInput.maxLength = 200;
        this.chatInput.style.cssText = `
            width: calc(100% - 20px);
            padding: 8px;
            margin: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #666;
            border-radius: 4px;
            color: white;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        `;
        
        // Assemble chat UI
        this.chatContainer.appendChild(this.chatMessages);
        this.chatContainer.appendChild(this.chatInput);
        document.body.appendChild(this.chatContainer);
        
        // Show initial help message
        this.displayMessage('System', 'Chat opened! Type /help for available commands.', '#00FF00');
    }
    
    setupEventHandlers() {
        // Global key handler
        document.addEventListener('keydown', (e) => {
            // Toggle chat with Enter key (when not focused on input)
            if (e.key === 'Enter' && document.activeElement !== this.chatInput) {
                e.preventDefault();
                this.toggleChat();
                return;
            }
            
            // Close chat with Escape
            if (e.key === 'Escape') {
                e.preventDefault();
                this.hideChat();
                return;
            }
            
            // Handle chat input
            if (this.isVisible && document.activeElement === this.chatInput) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.sendMessage();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    this.hideChat();
                }
            }
        });
        
        // Prevent chat input from interfering with game controls
        this.chatInput.addEventListener('keydown', (e) => {
            e.stopPropagation(); // Prevent key events from reaching game
        });
        
        this.chatInput.addEventListener('keyup', (e) => {
            e.stopPropagation(); // Prevent key events from reaching game
        });
        
        // Auto-hide chat after inactivity
        let hideTimeout;
        const resetHideTimer = () => {
            clearTimeout(hideTimeout);
            hideTimeout = setTimeout(() => {
                if (this.isVisible && document.activeElement !== this.chatInput) {
                    this.hideChat();
                }
            }, 10000); // Hide after 10 seconds of inactivity
        };
        
        this.chatContainer.addEventListener('mousemove', resetHideTimer);
        this.chatInput.addEventListener('focus', () => clearTimeout(hideTimeout));
        this.chatInput.addEventListener('blur', resetHideTimer);
    }
    
    toggleChat() {
        if (this.isVisible) {
            this.hideChat();
        } else {
            this.showChat();
        }
    }
    
    showChat() {
        this.isVisible = true;
        this.chatContainer.style.display = 'block';
        this.chatInput.focus();
        this.scrollToBottom();
    }
    
    hideChat() {
        this.isVisible = false;
        this.chatContainer.style.display = 'none';
        this.chatInput.blur();
    }
    
    sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Clear input
        this.chatInput.value = '';
        
        // Send to server
        if (this.networkManager) {
            this.networkManager.send(MessageType.PLAYER_TEXT, { text: message });
        }
        
        // Add to local history for display (will be echoed back by server)
        this.chatHistory.push({ sender: 'You', message, color: '#CCCCCC' });
        if (this.chatHistory.length > this.maxHistory) {
            this.chatHistory.shift();
        }
        
        // Don't hide chat immediately after sending message
    }
    
    displayMessage(sender, message, color = '#FFFFFF') {
        // Add to history
        this.chatHistory.push({ sender, message, color });
        if (this.chatHistory.length > this.maxHistory) {
            this.chatHistory.shift();
        }
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            margin-bottom: 4px;
            line-height: 1.4;
        `;
        
        // Format message with sender
        const senderSpan = document.createElement('span');
        senderSpan.style.color = color;
        senderSpan.style.fontWeight = 'bold';
        senderSpan.textContent = `[${sender}] `;
        
        const messageSpan = document.createElement('span');
        messageSpan.style.color = '#FFFFFF';
        messageSpan.textContent = message;
        
        messageDiv.appendChild(senderSpan);
        messageDiv.appendChild(messageSpan);
        
        // Add to chat messages
        this.chatMessages.appendChild(messageDiv);
        
        // Remove old messages if too many
        while (this.chatMessages.children.length > this.maxHistory) {
            this.chatMessages.removeChild(this.chatMessages.firstChild);
        }
        
        // Auto-scroll to bottom
        this.scrollToBottom();
        
        // Auto-show chat for new messages (but don't focus input)
        if (!this.isVisible) {
            this.chatContainer.style.display = 'block';
            setTimeout(() => {
                if (document.activeElement !== this.chatInput) {
                    this.chatContainer.style.display = 'none';
                }
            }, 5000); // Hide after 5 seconds if not actively using chat
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    /**
     * Add a system message (local only)
     */
    addSystemMessage(message, color = '#FFFF00') {
        this.displayMessage('System', message, color);
    }
    
    /**
     * Get current chat history
     */
    getChatHistory() {
        return this.chatHistory;
    }
    
    /**
     * Clear chat history
     */
    clearHistory() {
        this.chatHistory = [];
        this.chatMessages.innerHTML = '';
        this.addSystemMessage('Chat history cleared.');
    }
}

// Initialize chat system when DOM is ready
let chatSystem = null;

export function initializeChatSystem(networkManager) {
    if (!chatSystem && networkManager) {
        chatSystem = new ChatSystem(networkManager);
        
        // Expose globally for debugging
        window.chatSystem = chatSystem;
        
        console.log('[ChatSystem] Chat system ready. Press Enter to open chat.');
    }
    return chatSystem;
}

export function getChatSystem() {
    return chatSystem;
}