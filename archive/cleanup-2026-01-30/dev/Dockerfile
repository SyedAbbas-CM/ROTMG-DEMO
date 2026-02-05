FROM node:20-slim

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install --production

# Copy the rest of the application
COPY . .

# Expose the game server port
EXPOSE 4000

# Set environment variables
ENV NODE_ENV=production
ENV PORT=4000

# Start the server
CMD ["node", "Server.js"]
