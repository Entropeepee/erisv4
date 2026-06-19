import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# NOTE: You will need to set your API Key here!
# Example: os.environ.get("OPENAI_API_KEY")
API_KEY = "YOUR_API_KEY_HERE"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Example: Simple echo response for now if API key is not set
    if API_KEY == "YOUR_API_KEY_HERE":
        response_text = f"(Eris LLM is offline. Please add your API key to eris_llm_server.py). You said: {user_input}"
        return jsonify({"response": response_text})

    # Example OpenAI API Call (Uncomment and install openai if needed)
    """
    import openai
    openai.api_key = API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are Eris, a mystical AI avatar in a Dream World. Keep your responses short and mysterious."},
            {"role": "user", "content": user_input}
        ]
    )
    response_text = response.choices[0].message.content.strip()
    """
    return jsonify({"response": "LLM Integration Placeholder. Configure your LLM API to connect here!"})

if __name__ == '__main__':
    print("Eris LLM Server running on port 5000...")
    app.run(port=5000)
