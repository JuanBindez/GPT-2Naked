from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/puregpt2/api/v1/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        context = data.get("context", "")

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=50, 
                num_return_sequences=3, 
                temperature=0.7, 
                do_sample=True,
                top_k=50,
                num_beams=1,  # stop early_stopping
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
