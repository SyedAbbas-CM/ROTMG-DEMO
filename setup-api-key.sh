#!/bin/bash
# Quick setup script for Gemini API key

echo "========================================="
echo "  Gemini API Key Setup for ROTMG-DEMO"
echo "========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp env.example.txt .env
    echo "✓ .env file created"
    echo ""
fi

# Check if API key is already set
if grep -q "GOOGLE_API_KEY=.\+" .env 2>/dev/null; then
    echo "✓ API key already configured in .env"
    echo ""
    echo "Testing current API key..."
    node test-gemini-api.js
    exit 0
fi

echo "No API key found. Let's set one up!"
echo ""
echo "Opening Google AI Studio in your browser..."
echo "URL: https://aistudio.google.com/app/apikey"
echo ""

# Try to open the browser (macOS)
open "https://aistudio.google.com/app/apikey" 2>/dev/null || {
    echo "Please open this URL manually:"
    echo "https://aistudio.google.com/app/apikey"
}

echo ""
echo "Steps to get your API key:"
echo "1. Sign in with your Google account"
echo "2. Click 'Create API key'"
echo "3. Copy the generated key"
echo ""
read -p "Paste your API key here: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "❌ No API key entered. Exiting."
    exit 1
fi

# Update .env file
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/GOOGLE_API_KEY=.*/GOOGLE_API_KEY=$API_KEY/" .env
else
    # Linux
    sed -i "s/GOOGLE_API_KEY=.*/GOOGLE_API_KEY=$API_KEY/" .env
fi

echo ""
echo "✓ API key saved to .env file"
echo ""
echo "Testing API connection..."
node test-gemini-api.js

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "  ✅ Setup complete!"
    echo "========================================="
    echo ""
    echo "You can now start the server with:"
    echo "  npm start"
    echo ""
    echo "The boss AI will use Gemini to make decisions!"
else
    echo ""
    echo "❌ API test failed. Please check your API key."
    echo ""
    echo "To try again, run: ./setup-api-key.sh"
fi
