from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        if request.method == 'POST':
            Text_to_detect_stress = str(request.form['Enter a text'])
            pre_process_data = 'Pre_processing.pkl'
            model_file = 'finalized_model.pkl'
            scaler = pickle.load(open(pre_process_data, 'rb'))
            model = pickle.load(open(model_file, 'rb'))
            test = scaler.transform([Text_to_detect_stress])
            prediction = str(model.predict(test))
            print(type(prediction))
            print("prediction is ", prediction)
            return render_template('results.html', prediction=prediction)
    except Exception as e:
        print("The Error is ",str(e))

if __name__ == "__main__" :
    app.run(debug=True,port=5006)
