from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

log_model = pickle.load(open('iris_classifier.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/multiply', methods=['POST'])
def multiply_numbers():

    global log_model

    data = request.get_json()
    input_data = data['flower_in']
    int_input_data = [float(val) for val in input_data]

    prediction = log_model.predict([int_input_data])

    if prediction[0]==0:
        return jsonify({'result': "SETOSA"})
    elif prediction[0]==1:
        return jsonify({'result': "VERSICOLOR"})
    else:
        return jsonify({'result': "VIRGINICA"})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=6969)
