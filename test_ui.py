from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Mock prediction
    is_dos = data.get('rate', 0) > 1000 or data.get('sttl', 0) > 200
    cat = 'DoS' if is_dos else 'Normal'
    probs = {'Normal': 0.05, 'DoS': 0.95, 'Reconnaissance': 0.0, 'Fuzzer': 0.0} if is_dos else {'Normal': 0.99, 'DoS': 0.0, 'Reconnaissance': 0.0, 'Fuzzer': 0.0}
    return jsonify({
        'predicted_category': cat,
        'prediction_probability_for_category': probs[cat],
        'all_class_probabilities': probs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
