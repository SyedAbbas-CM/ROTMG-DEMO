/**
 * Chat panel component for displaying and sending messages
 */
import { UIComponent } from '../UIManager.js';

export class ChatPanel extends UIComponent {
  /**
   * Create a chat panel component
   * @param {Object} gameState - Game state reference
   * @param {Object} manager - UI manager reference
   */
  constructor(gameState, manager) {
    super(gameState, manager);
    
    this.messages = [];
    this.channels = ['All', 'Party', 'Guild', 'Trade'];
    this.currentChannel = 'All';
  }
  
  /**
   * Initialize the component
   * @returns {HTMLElement} The component's DOM element
   */
  async init() {
    // Create panel container
    this.element = document.createElement('div');
    this.element.className = 'ui-chat-panel';
    this.element.style.position = 'absolute';
    this.element.style.bottom = '48px';
    this.element.style.left = '16px';
    this.element.style.width = '320px';
    this.element.style.height = '180px';
    this.element.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    this.element.style.border = '2px solid #444';
    this.element.style.color = 'white';
    this.element.style.zIndex = '10';
    this.element.style.pointerEvents = 'auto';
    this.element.style.display = 'flex';
    this.element.style.flexDirection = 'column';
    
    // Create panel header
    const header = document.createElement('div');
    header.className = 'panel-header';
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.padding = '2px 8px';
    header.style.borderBottom = '1px solid #333';
    header.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    
    const title = document.createElement('div');
    title.className = 'panel-title';
    title.textContent = 'Chat';
    title.style.fontSize = '12px';
    title.style.color = '#aaa';
    
    const controls = document.createElement('div');
    controls.className = 'panel-controls';
    controls.style.display = 'flex';
    
    const minimizeBtn = document.createElement('button');
    minimizeBtn.className = 'btn-minimize';
    minimizeBtn.innerHTML = '&minus;';
    minimizeBtn.title = 'Minimize';
    minimizeBtn.style.background = 'none';
    minimizeBtn.style.border = 'none';
    minimizeBtn.style.color = '#666';
    minimizeBtn.style.cursor = 'pointer';
    minimizeBtn.style.fontSize = '12px';
    minimizeBtn.style.width = '16px';
    minimizeBtn.style.height = '16px';
    minimizeBtn.style.display = 'flex';
    minimizeBtn.style.alignItems = 'center';
    minimizeBtn.style.justifyContent = 'center';
    minimizeBtn.style.marginLeft = '4px';
    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.title = 'Close';
    closeBtn.style.background = 'none';
    closeBtn.style.border = 'none';
    closeBtn.style.color = '#666';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.fontSize = '12px';
    closeBtn.style.width = '16px';
    closeBtn.style.height = '16px';
    closeBtn.style.display = 'flex';
    closeBtn.style.alignItems = 'center';
    closeBtn.style.justifyContent = 'center';
    closeBtn.style.marginLeft = '4px';
    
    controls.appendChild(minimizeBtn);
    controls.appendChild(closeBtn);
    
    header.appendChild(title);
    header.appendChild(controls);
    
    // Chat tabs
    const tabs = document.createElement('div');
    tabs.className = 'chat-tabs';
    tabs.style.display = 'flex';
    tabs.style.borderBottom = '1px solid #333';
    
    this.channels.forEach(channel => {
      const tab = document.createElement('button');
      tab.className = `chat-tab ${channel === this.currentChannel ? 'active' : ''}`;
      tab.textContent = channel;
      tab.style.flex = '1';
      tab.style.backgroundColor = channel === this.currentChannel ? '#333' : 'transparent';
      tab.style.color = channel === this.currentChannel ? '#f59e0b' : '#666';
      tab.style.border = 'none';
      tab.style.borderRight = '1px solid #333';
      tab.style.padding = '4px 0';
      tab.style.fontSize = '11px';
      tab.style.cursor = 'pointer';
      
      tab.addEventListener('click', () => this.setChannel(channel));
      
      tabs.appendChild(tab);
    });
    
    // Chat content
    const content = document.createElement('div');
    content.className = 'chat-content';
    content.style.flex = '1';
    content.style.overflowY = 'auto';
    content.style.padding = '8px';
    content.style.fontSize = '12px';
    
    // Empty state message
    if (this.messages.length === 0) {
      const emptyMessage = document.createElement('div');
      emptyMessage.className = 'empty-chat';
      emptyMessage.textContent = 'No messages yet.';
      emptyMessage.style.color = '#666';
      emptyMessage.style.textAlign = 'center';
      emptyMessage.style.paddingTop = '20px';
      content.appendChild(emptyMessage);
    }
    
    // Chat input
    const inputContainer = document.createElement('div');
    inputContainer.className = 'chat-input-container';
    inputContainer.style.display = 'flex';
    inputContainer.style.padding = '4px 8px';
    inputContainer.style.borderTop = '1px solid #333';
    inputContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    
    const input = document.createElement('input');
    input.className = 'chat-input';
    input.type = 'text';
    input.placeholder = 'Type your message...';
    input.style.flex = '1';
    input.style.backgroundColor = '#222';
    input.style.border = '1px solid #333';
    input.style.color = 'white';
    input.style.padding = '4px 8px';
    input.style.fontSize = '12px';
    
    const sendBtn = document.createElement('button');
    sendBtn.className = 'chat-send';
    sendBtn.innerHTML = '&#10148;'; // Right arrow
    sendBtn.title = 'Send';
    sendBtn.style.backgroundColor = '#333';
    sendBtn.style.border = '1px solid #444';
    sendBtn.style.borderLeft = 'none';
    sendBtn.style.color = '#f59e0b';
    sendBtn.style.cursor = 'pointer';
    sendBtn.style.width = '24px';
    sendBtn.style.display = 'flex';
    sendBtn.style.alignItems = 'center';
    sendBtn.style.justifyContent = 'center';
    
    inputContainer.appendChild(input);
    inputContainer.appendChild(sendBtn);
    
    // Setup event handlers
    minimizeBtn.addEventListener('click', () => this.minimize());
    closeBtn.addEventListener('click', () => this.hide());
    
    // Keep track of chat input focus state globally
    window.chatInputActive = false;
    
    // Handle focus and blur events to toggle the chat input state
    input.addEventListener('focus', () => {
      window.chatInputActive = true;
      input.style.backgroundColor = '#333'; // Darken background to indicate focus
      input.style.borderColor = '#f59e0b'; // Add orange border when focused
      console.log('Chat input focused - game controls temporarily disabled');
    });
    
    input.addEventListener('blur', () => {
      window.chatInputActive = false;
      input.style.backgroundColor = '#222'; // Reset to default
      input.style.borderColor = '#333'; // Reset to default
      console.log('Chat input blurred - game controls restored');
    });
    
    // Stop propagation of all key events when input is focused to prevent game controls from triggering
    input.addEventListener('keydown', (e) => {
      // Always stop propagation when chat input is focused
      e.stopPropagation();
      
      if (e.key === 'Enter' && input.value.trim()) {
        this.sendMessage(input.value.trim());
        input.value = '';
        
        // Keep focus in the input field after sending a message
        // Do not blur here to allow continuous chatting
      }
      
      // Handle the Escape key to exit chat input
      if (e.key === 'Escape') {
        input.blur();
      }
    });
    
    // Also prevent key up events from reaching the game
    input.addEventListener('keyup', (e) => {
      e.stopPropagation();
    });
    
    // Also prevent keypress events
    input.addEventListener('keypress', (e) => {
      e.stopPropagation();
    });
    
    sendBtn.addEventListener('click', () => {
      if (input.value.trim()) {
        this.sendMessage(input.value.trim());
        input.value = '';
        
        // Keep focus in the input field after sending
        input.focus();
      }
    });
    
    // Append elements to panel
    this.element.appendChild(header);
    this.element.appendChild(tabs);
    this.element.appendChild(content);
    this.element.appendChild(inputContainer);
    
    // Save references
    this.chatContent = content;
    this.chatInput = input;
    
    // Add initial messages if any
    this.renderMessages();
    
    // Initialize network listeners
    setTimeout(() => {
      this.setupNetworkListeners();
    }, 1000); // Short delay to ensure network manager is ready
    
    return this.element;
  }
  
  /**
   * Minimize the panel
   */
  minimize() {
    this.hide();
    this.manager.trigger('minimizeComponent', { id: 'chatPanel' });
  }
  
  /**
   * Set the active channel
   * @param {string} channel - Channel name
   */
  setChannel(channel) {
    if (this.channels.includes(channel)) {
      this.currentChannel = channel;
      
      // Update tab visuals
      const tabs = this.element.querySelectorAll('.chat-tab');
      tabs.forEach(tab => {
        if (tab.textContent === channel) {
          tab.style.backgroundColor = '#333';
          tab.style.color = '#f59e0b';
          tab.classList.add('active');
        } else {
          tab.style.backgroundColor = 'transparent';
          tab.style.color = '#666';
          tab.classList.remove('active');
        }
      });
      
      // Re-render messages filtered for this channel
      this.renderMessages();
    }
  }
  
  /**
   * Send a chat message
   * @param {string} message - Message text
   */
  sendMessage(message) {
    // Generate a unique ID for this message
    const messageId = Date.now() + Math.floor(Math.random() * 1000);
    
    // Send to server if connected
    if (this.gameState.networkManager && this.gameState.networkManager.isConnected()) {
      // Send chat message to server using the dedicated method
      if (this.gameState.networkManager.sendChatMessage) {
        this.gameState.networkManager.sendChatMessage({
          id: messageId, 
          message: message,
          channel: this.currentChannel,
          timestamp: Date.now(),
          clientId: this.gameState.clientId
        });
      } else {
        // Fallback to generic send method
        this.gameState.networkManager.send('chat', {
          id: messageId,
          message: message,
          channel: this.currentChannel,
          timestamp: Date.now(),
          clientId: this.gameState.clientId
        });
      }
      
      console.log('Sent chat message to server:', message);
      
      // No local message is displayed - we'll wait for the server to send it back
    } else {
      // Not connected - show an error message
      this.addMessage({
        text: 'Cannot send message: Not connected to server',
        sender: 'System',
        type: 'error',
        channel: 'All'
      });
      
      console.warn('Cannot send chat message: Not connected to server');
    }
  }
  
  /**
   * Initialize network listeners for chat messages
   */
  setupNetworkListeners() {
    // Skip if network manager is not available
    if (!this.gameState.networkManager) {
      console.warn('Network manager not available for chat');
      return;
    }
    
    // Check if the network manager has the 'on' method
    if (typeof this.gameState.networkManager.on !== 'function') {
      console.warn('Network manager does not have an "on" method for registering chat handlers');
      return;
    }
    
    try {
      // Import MessageType from the network manager
      let MessageType;
      if (window.MessageType) {
        MessageType = window.MessageType;
      } else {
        // Hardcode the value if we can't import it
        MessageType = { CHAT_MESSAGE: 90 };
        console.warn('Using hardcoded MessageType.CHAT_MESSAGE value (90)');
      }
      
      // Register for binary CHAT_MESSAGE (type 90) from server
      this.gameState.networkManager.on(MessageType.CHAT_MESSAGE, (data) => {
        // Process the incoming chat message
        this.receiveChatFromServer(data);
      });
      
      console.log('Chat network listeners initialized for message type: ' + MessageType.CHAT_MESSAGE);
    } catch (error) {
      console.error('Error setting up chat network listeners:', error);
    }
  }
  
  /**
   * Process an incoming chat message from the server
   * @param {Object} data - Message data from server
   */
  receiveChatFromServer(data) {
    if (!data || !data.message) {
      console.warn('Received invalid chat data from server');
      return;
    }
    
    // Log detailed diagnostic info
    console.log('Chat source info:', {
      messageFromClientId: data.clientId,
      ourClientId: this.gameState.clientId,
      isSelf: data.isOwnMessage === true,
      senderName: data.sender,
      message: data.message
    });
    
    // Create message object from server data - use exact sender name from server
    const messageObj = {
      id: data.id,
      text: data.message,
      sender: data.sender || 'Unknown',
      channel: data.channel || 'All',
      type: data.type || 'player',
      timestamp: data.timestamp ? new Date(data.timestamp) : new Date(),
      clientId: data.clientId, // Keep clientId for styling our own messages
      isOwnMessage: data.isOwnMessage === true // Flag from server if this is our message
    };
    
    console.log('Received chat message from server:', messageObj);
    
    // Check if we already have this message (by ID) to avoid duplicates
    if (this.messages.some(msg => msg.id === data.id)) {
      console.log('Skipping duplicate message with ID:', data.id);
      return;
    }
    
    // Add to chat panel
    this.addMessage(messageObj);
  }
  
  /**
   * Add a message from the system or other players
   * @param {Object} message - Message object
   */
  addMessage(message) {
    // Ensure message has required properties
    const messageObj = {
      id: message.id || Date.now(),
      text: message.text || '',
      sender: message.sender || 'System',
      channel: message.channel || 'All',
      type: message.type || 'system',
      timestamp: message.timestamp || new Date()
    };
    
    // Add to messages array
    this.messages.push(messageObj);
    
    // Add message to UI if channel matches
    if (messageObj.channel === this.currentChannel || messageObj.channel === 'All' || this.currentChannel === 'All') {
      this.addMessageElement(messageObj);
    }
    
    return messageObj;
  }
  
  /**
   * Add a message element to the chat
   * @param {Object} message - Message object
   */
  addMessageElement(message) {
    // Remove empty state if present
    const emptyChat = this.chatContent.querySelector('.empty-chat');
    if (emptyChat) {
      this.chatContent.removeChild(emptyChat);
    }
    
    // Create message element
    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${message.type}`;
    messageEl.style.marginBottom = '4px';
    messageEl.style.wordBreak = 'break-word';
    
    // Check if this message is from the current user using the isOwnMessage flag
    const isOwnMessage = message.isOwnMessage === true;
    
    // Apply a special style for own messages
    if (isOwnMessage) {
      messageEl.style.textAlign = 'right';
      messageEl.style.paddingRight = '5px';
      messageEl.style.borderRight = '2px solid #f59e0b';
    }
    
    let messagePrefix;
    if (message.type === 'system') {
      messageEl.style.color = '#a3a3a3';
      messagePrefix = '[System]';
    } else if (message.type === 'player') {
      messageEl.style.color = 'white';
      messagePrefix = `[${message.sender}]`;
    } else if (message.type === 'party') {
      messageEl.style.color = '#60a5fa';
      messagePrefix = `[${message.sender}]`;
    } else if (message.type === 'guild') {
      messageEl.style.color = '#4ade80';
      messagePrefix = `[${message.sender}]`;
    } else if (message.type === 'trade') {
      messageEl.style.color = '#f59e0b';
      messagePrefix = `[${message.sender}]`;
    } else if (message.type === 'whisper') {
      messageEl.style.color = '#c084fc';
      messagePrefix = `From ${message.sender}`;
    }
    
    messageEl.innerHTML = `<span style="color: ${this.getColorForType(message.type)}">${messagePrefix}: </span>${message.text}`;
    
    // Add to chat content
    this.chatContent.appendChild(messageEl);
    
    // Scroll to bottom
    this.chatContent.scrollTop = this.chatContent.scrollHeight;
  }
  
  /**
   * Get color for message type
   * @param {string} type - Message type
   * @returns {string} Color hex
   */
  getColorForType(type) {
    switch (type) {
      case 'system': return '#a3a3a3';
      case 'player': return '#f5f5f5';
      case 'party': return '#60a5fa';
      case 'guild': return '#4ade80';
      case 'trade': return '#f59e0b';
      case 'whisper': return '#c084fc';
      case 'error': return '#f87171';
      case 'pending': return '#9ca3af'; // Gray for pending messages
      default: return '#f5f5f5';
    }
  }
  
  /**
   * Render all messages for the current channel
   */
  renderMessages() {
    // Clear current messages
    this.chatContent.innerHTML = '';
    
    // Filter messages for current channel
    const filteredMessages = this.messages.filter(msg => 
      msg.channel === this.currentChannel || 
      msg.channel === 'All' || 
      this.currentChannel === 'All'
    );
    
    // Show empty state if no messages
    if (filteredMessages.length === 0) {
      const emptyMessage = document.createElement('div');
      emptyMessage.className = 'empty-chat';
      emptyMessage.textContent = 'No messages yet.';
      emptyMessage.style.color = '#666';
      emptyMessage.style.textAlign = 'center';
      emptyMessage.style.paddingTop = '20px';
      this.chatContent.appendChild(emptyMessage);
      return;
    }
    
    // Add each message
    filteredMessages.forEach(message => {
      this.addMessageElement(message);
    });
  }
  
  /**
   * Update the component with game state
   * @param {Object} gameState - Current game state
   */
  update(gameState) {
    // Check if we have new client ID information
    if (gameState.clientId && (!this.gameState.clientId || this.gameState.clientId !== gameState.clientId)) {
      this.gameState.clientId = gameState.clientId;
      console.log('Chat panel updated with client ID:', this.gameState.clientId);
    }
    
    // Handle new messages from game state if needed
    if (gameState.newMessages && gameState.newMessages.length > 0) {
      gameState.newMessages.forEach(message => {
        this.addMessage(message);
      });
      
      // Clear processed messages
      gameState.newMessages = [];
    }
  }
} 