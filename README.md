# 🔋 EV Battery Predictive Maintenance System

An end-to-end machine learning pipeline and interactive dashboard designed to analyze real-time Electric Vehicle (EV) battery telemetry and predict End of Life (EOL) degradation.

## 🚀 Project Overview
This system uses a machine learning backend to process critical battery parameters—such as internal resistance, charge cycles, cell temperature, and voltage—to determine if a battery pack is healthy or requires immediate replacement. 

### Key Features
* **Real-Time Diagnostics:** Live inference via a custom Python API.
* **Machine Learning Engine:** Powered by a Scikit-Learn Random Forest Classifier utilizing democratic voting for high-accuracy predictions.
* **Confidence Scoring:** Outputs mathematical certainty percentages for every diagnostic run.
* **Fleet Simulation:** Dynamically loads or generates realistic degraded vehicle telemetry for testing.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn, Pandas, Joblib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

## ⚙️ Local Setup & Deployment
To run this application locally on a Windows machine:

1.Set up the virtual environment:
-> DOS
py -3.12 -m venv env
env\Scripts\activate

2.Install dependencies:
-> DOS
pip install fastapi "uvicorn[standard]" pandas joblib scikit-learn==1.6.1

3.Start the AI Server:
DOS
uvicorn api:app --reload

4.Launch the Dashboard:
Open index.html in any modern web browser to access the EOL Diagnostics Console.
