# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
import psycopg2
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
DB_CONFIG = {
    'dbname': 'demo',
    'user': 'postgres',
    'password': 'tff',
    'host': 'localhost',
    'port': '5432'
}

# Predefined users
USERS = {
    'Admin': 'Admin@2003'
}


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Existing queries
    cur.execute("""
        SELECT age, height, weight, duration, heart_rate, 
               body_temp, gender, calories, prediction_date 
        FROM predictions 
        ORDER BY prediction_date DESC LIMIT 5
    """)
    recent_predictions = cur.fetchall()
    
    cur.execute("SELECT COALESCE(AVG(calories), 0) FROM predictions")
    avg_calories = round(cur.fetchone()[0], 2)
    
    cur.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cur.fetchone()[0]
    
    # New queries for gender distribution
    cur.execute("SELECT COUNT(*) FROM predictions WHERE gender = 1")
    male_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM predictions WHERE gender = 0")
    female_count = cur.fetchone()[0]
    
    conn.close()
    
    return render_template('dashboard.html', 
                         recent_predictions=recent_predictions,
                         avg_calories=avg_calories,
                         total_predictions=total_predictions,
                         male_count=male_count,
                         female_count=female_count)
    
# @app.route('/dashboard')
# @login_required
# def dashboard():
#     conn = psycopg2.connect(**DB_CONFIG)
#     cur = conn.cursor()
    
#     cur.execute("""
#         SELECT age, height, weight, duration, heart_rate, 
#                body_temp, gender, calories, prediction_date 
#         FROM predictions 
#         ORDER BY prediction_date DESC LIMIT 5
#     """)
#     recent_predictions = cur.fetchall()
    
#     cur.execute("SELECT COALESCE(AVG(calories), 0) FROM predictions")
#     avg_calories = round(cur.fetchone()[0], 2)
    
#     cur.execute("SELECT COUNT(*) FROM predictions")
#     total_predictions = cur.fetchone()[0]
    
#     conn.close()
    
#     return render_template('dashboard.html', 
#                          recent_predictions=recent_predictions,
#                          avg_calories=avg_calories,
#                          total_predictions=total_predictions)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])
        gender = 1 if request.form['gender'] == 'Male' else 0
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        bmi = round(bmi, 2)
        
        # Prepare data for prediction
        data = {
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Body_Temp': [body_temp],
            'BMI': [bmi],
            'Gender_male': [gender]
        }
        df = pd.DataFrame(data)
        
        # Load and prepare training data
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")
        exercise_df = exercise.merge(calories, on="User_ID")
        exercise_df.drop(columns="User_ID", inplace=True)
        
        # Train model
        exercise_train_data, _ = train_test_split(exercise_df, test_size=0.2, random_state=1)
        exercise_train_data['BMI'] = exercise_train_data['Weight'] / ((exercise_train_data['Height'] / 100) ** 2)
        exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
        
        X_train = exercise_train_data.drop("Calories", axis=1)
        y_train = exercise_train_data["Calories"]
        
        model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
        model.fit(X_train, y_train)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Save prediction to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (age, height, weight, duration, heart_rate, body_temp, gender, calories)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (age, height, weight, duration, heart_rate, body_temp, gender, prediction))
        conn.commit()
        conn.close()
        
        return render_template('predict.html', prediction=round(prediction, 2))
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)