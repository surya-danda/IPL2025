from flask import * 
import numpy as np
import pickle
import pandas as pd

app=Flask(__name__)

model= pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    # Collect inputs
    runs = float(request.form['runs'])
    wickets = float(request.form['wickets'])
    overs = float(request.form['overs'])
    runs_last5 = float(request.form['runs_last5'])
    wickets_last5 = float(request.form['wickets_last5'])
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']


    def team():
        # Create a dictionary for input features
        input_data = {
            'runs': runs,
            'wickets': wickets,
            'overs': overs,
            'runs_last5': runs_last5,
            'wickets_last5': wickets_last5,
            'batting_team': batting_team,
            'bowling_team': bowling_team
        }

        # Encode categorical features using pd.get_dummies()
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)

        # Align the columns with the training features
        train_columns = model.feature_names_in_  # Retrieve feature names from the model
        input_aligned = input_encoded.reindex(columns=train_columns, fill_value=0)

        # Convert to numpy array for prediction
        features = input_aligned.to_numpy()

        # Predict using the trained model
        prediction = model.predict(features)
        return render_template(
            "index.html",
            prediction_text=f"The predicted score is {int(prediction) } - {int(prediction+6 )} runs"
        )

    valid_batting_teams = [
        'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders', 'Mumbai Indians',
        'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
        'Gujarat Titans', 'Lucknow Super Giants'
    ]

    valid_bowling_teams = [
        'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders', 'Mumbai Indians',
        'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
        'Gujarat Titans', 'Lucknow Super Giants'
    ]
    if runs < 0 or runs > 300:
        return render_template(
            "index.html",
            error="Please enter valid runs(guru idhi t20 matchuuu odi kadhu...)."
        )    
    if wickets < 0 or wickets > 9:
        return render_template(
            "index.html",
            error=f"Please enter valid no.of wickets , you entered {int(wickets)} wickets which is not valid"
        )   
    if overs < 1 or overs > 20:
        return render_template(
            "index.html",
            error=f"Please enter valid no.of overs , you entered {int(overs)} overs which is not valid"
        )   
    if runs_last5 < 0 or runs_last5 > 180:
        return render_template(
            "index.html",
            error="Please enter valid runs(over ki 6 sixes kottina 5 overs ki 180 eh kodatharu guru)"
        )   
    if wickets_last5 < 0 or wickets_last5 > 9:
        return render_template(
            "index.html",
            error=f"Please enter valid no.of last 5 overs wickets , you entered {int(wickets)} wickets which is not valid "
        )
    else:
        return team()
    
    
    
   





if __name__=="__main__":
    app.run(debug=True)