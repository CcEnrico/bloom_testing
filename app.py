from flask import Flask, request, jsonify
from resources import load_resources
from query_utils import retrieve_context, generate_response

# Initialize Flask app
app = Flask(__name__)

# Load resources globally during app initialization
print("Initializing resources...")
tokenizer, model, device, index, documents, embedding_model = load_resources()
print("Resources initialized successfully!")



@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    # Retrieve context and generate response
    context = retrieve_context(query_text, index=index, documents=documents, embedding_model=embedding_model)
    response = generate_response(context, query_text, tokenizer=tokenizer, model=model, device=device)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
