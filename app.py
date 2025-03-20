from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load saved model
model = pickle.load(open('model/gwp.pkl', 'rb'))

# Mapping dictionaries (these must match your training data encoding)
quarter_mapping = {'Quarter1': 0, 'Quarter2': 1, 'Quarter3': 2, 'Quarter4': 3}
department_mapping = {'sewing': 1, 'finishing': 0}
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Saturday': 4, 'Sunday': 5}
team_mapping = {'Team 1': 1, 'Team 2': 2, 'Team 3': 3, 'Team 4': 4, 'Team 5': 5}
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect input from form
        quarter_input = request.form['quarter']
        department_input = request.form['department']
        day_input = request.form['day']
        team_input = request.form['team']
        month_input = request.form['month']

        # Convert categorical to numerical using mapping
        quarter = quarter_mapping.get(quarter_input, -1)
        department = department_mapping.get(department_input, -1)
        day = day_mapping.get(day_input, -1)
        team = team_mapping.get(team_input, -1)
        month = month_mapping.get(month_input, -1)

        # If any mapping failed (invalid input), return error
        if -1 in [quarter, department, day, team, month]:
            return render_template('submit.html', prediction_text='Error: Invalid categorical input.')

        # Collect numeric inputs
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        wip = float(request.form['wip'])
        overtime = float(request.form['overtime'])
        incentive = float(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = float(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = int(request.form['no_of_workers'])

        # Prepare feature array (in correct order)
        final_features = [[quarter, department, day, team, targeted_productivity,
                           smv, wip, overtime, incentive, idle_time,
                           idle_men, month, no_of_style_change, no_of_workers]]

        # Convert to numpy array
        final_features = np.array(final_features)

        # Predict
        prediction = model.predict(final_features)

        return render_template('submit.html', prediction_text=f'Predicted Productivity: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
