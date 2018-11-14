from flask import Flask, request, abort, jsonify
from predict import predict_unseen_data
import json

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def read_data():
    return jsonify({'hi': 'hi'})
    if not request.get_json():
        abort(400)
    json_data = request.get_json
    json_data = json.dumps(json_data)
    correctness_rate = predict_unseen_data(json_data)
    return jsonify({'correctness_rate': correctness_rate})


if __name__ == '__main__':
    app.run()