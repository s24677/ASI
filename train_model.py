import pandas as pd
import numpy as np

def loadData(df):
    df = df.drop(axis=1, labels='Patient_ID')

    # Convert "Low", "Medium", "High" values to numerical data

    type_map = {
        'Breast': 0,
        'Prostate': 1,
        'Skin': 2,
        'Colon': 3,
        'Lung': 4
    }

    risk_map = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }

    df.loc[:, 'Cancer_Type'] = df['Cancer_Type'].map(type_map)
    df.loc[:, 'Risk_Level'] = df['Risk_Level'].map(risk_map)

    return(df)

def train(df):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.3)

    features = df.columns.to_list()
    features = features[1:]
    print(df)

    X_train = train.loc[:, features] # the TRAINING dataset with explanatory features
    y_train = train['Cancer_Type'] # the TRAINING set with the feature we want to learn to predict. 

    X_test = test.loc[:, features] # the TEST set with the explanatory features 
    y_test= test['Cancer_Type'] # the TEST set with the feature we want to learn to predict.  


    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X_train,y_train)

    predictions = model.predict(X_test)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print('RMSE = ', rmse)

    import joblib

    joblib.dump(model, 'model.joblib')
    print('...Model saved...')

    return rmse

def predict(input_data):
    model = joblib.load('model.joblib')
    prediction = model.predict(input_data)
    return int(np.round(prediction[0]))
