from flask import Flask,render_template,request
import numpy as np
import joblib
import pickle
print('here3')
app=Flask(__name__)
print('here4')


    
with open('scaler.pkl', 'rb') as handle:
    scaler = pickle.load(handle)

with open('clf_bin.pkl', 'rb') as handle:
    _bin = pickle.load(handle)
    
with open('clf_reg.pkl', 'rb') as handle:
    reg = pickle.load(handle)
   

@app.route("/")
def home():
    print('here1')
   # return tf_idf,log_reg,naive_bayes}
    return render_template("main.html",output=False)

@app.route('/mainResult',methods=['POST'])
def prediction():
     input_list_name=['Cycle',
                      'SensorMeasure2',
                      'SensorMeasure3',
                      'SensorMeasure4',
                      'SensorMeasure7',
                      'SensorMeasure8',
                      'SensorMeasure9',
                      'SensorMeasure11',
                      'SensorMeasure12', 
                      'SensorMeasure13',
                      'SensorMeasure14', 
                      'SensorMeasure15', 
                      'SensorMeasure17',
                      'SensorMeasure20',
                      'SensorMeasure21']
     
     input_list_value=[]
     if request.method == 'POST':
         for i,name in enumerate(input_list_name):
            input_list_value.append(request.form[name])
     
     print(input_list_value)
     print(type(input_list_value))
     
     
    
     a = np.array(input_list_value)
     a=np.reshape(a,(1, a.size))
     a= scaler.transform(a)
    
     
     
     RUL=reg.predict(a)
     print(RUL)
    
     
    
     RUL_Binary=_bin.predict(a)
     mapping=lambda x: "Engine Is Okay" if x==1 else "Engine Is Not Okay"
     engine_health=mapping(RUL_Binary)
     
     return render_template('main.html',output=True,RemainingUL=RUL,condition=engine_health)

if __name__ == '__main__':
    print('here2')
    app.run()
