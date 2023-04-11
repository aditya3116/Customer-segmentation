from flask import Flask,render_template,request
import pickle
with open("mall.pkl","rb") as file:
    model=pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Get the  data from post method
    data=request.form.to_dict()
    data=[[int(data['Annual Income']),
    int(data['Spending Score'])]]
    
    # make predictions
    prediction =model.predict(data)
    if prediction==0:
        return "Careless Customer"
    elif prediction==1:
        return "Standard Customer"
    elif prediction==2:
        return "Target Customer"
    elif prediction==3:
        return "Sensible Customer"
    else:
        return "Careful Customer"
#return the predictions
    return str(prediction[0])
    return render_template('result.html')
    
if __name__=="__main__":
    app.run(port=8000,debug=True)

