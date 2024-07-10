from flask import Flask, request, jsonify
import traceback

from inference import predict

app = Flask(__name__)


@app.route('/video_qa', methods=['POST'])
def video_qa():
    if 'video' not in request.files:
        return jsonify({'error': 'no video file found'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'no chosen file'}), 400

    if 'question' not in request.form:
        question = ""
    else:
        question = request.form['question']

    if question is None or question == "" or question == "@Caption":
        question = "Please describe the video in detail."

    print("Get question:", question)

    if 'temperature' not in request.form:
        temperature = 0.001
        print("No temperature found, use default value 0.001")
    else:
        temperature = float(request.form['temperature'])
        print("Get temperature:", temperature)

    try:
        answer = predict(prompt=question, video_data=video.read(), temperature=temperature)
        return jsonify(
            {"answer": answer})
    except:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
