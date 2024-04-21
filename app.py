from flask import Flask, request, jsonify # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from flask_cors import CORS  # type: ignore

app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    # Get user input from the request
    user_input = request.json["message"]

    # Tokenize the input text
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate response
    response_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

    # Decode the generated response, excluding the user's input
    bot_response = tokenizer.decode(response_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    # Return the bot's response as JSON
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
