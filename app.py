from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            cp=request.form.get('cp'),
            trtbs=request.form.get('trtbps'),
            chol=request.form.get('chol'),
            fbs=request.form.get('fbs'),
            restecg=request.form.get('restecg'),
            thalachh=request.form.get('thalachh'),
            exng=request.form.get('exng'),
            oldpeak=request.form.get('oldpeak'),
            slp=request.form.get('slp'),
            caa=request.form.get('caa'),
            thall=request.form.get('thall')
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
