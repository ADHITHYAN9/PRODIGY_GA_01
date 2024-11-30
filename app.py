from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model_name = "D:/GEN AI PROJECT/fine_tuned_gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route("/")
def home():
    # Serve the index.html page
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_text():
    # Get the prompt and parameters from the frontend
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_length = int(data.get("max_length", 100))  # Default max_length to 100 if not provided
    temperature = float(data.get("temperature", 0.7))  # Default temperature to 0.7 if not provided

    if not prompt:
        return jsonify({"error": "Prompt is required!"}), 400

    # Tokenize the input and generate text with the specified parameters
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text based on the input prompt and parameters
    output = model.generate(input_ids, 
                            max_length=max_length, 
                            num_return_sequences=1, 
                            no_repeat_ngram_size=2, 
                            top_k=50, 
                            top_p=0.95, 
                            temperature=temperature)

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(debug=True)
