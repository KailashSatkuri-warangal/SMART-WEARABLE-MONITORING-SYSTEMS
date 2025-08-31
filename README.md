# AI in Healthcare – Fitbit Data Analysis  

This project leverages **AI and Machine Learning** to analyze **Fitbit health data** collected over multiple months. The system preprocesses raw health data, builds predictive models, and evaluates health outcomes with the goal of demonstrating how wearable devices can aid in healthcare research and decision-making.  

## 📂 Repository Structure  
```
.
├── .devcontainer/                # Dev container setup
├── mturkfitbit_export_3.12.16-4.11.16/   # Raw Fitbit dataset (batch 1)
├── mturkfitbit_export_4.12.16-5.12.16/   # Raw Fitbit dataset (batch 2)
├── static/                       # Static files for web app
├── app.py                        # Main Flask/Streamlit app
├── app1.py                       # Alternate app implementation
├── db.py                         # Database connection (SQLite)
├── eval.py                       # Model evaluation script
├── preprocessig.py               # Data preprocessing logic
├── health_data.db                # Sample health database
├── merged_fitbit_dataset.csv     # Preprocessed dataset
├── rf_model.pkl                  # Random Forest trained model
├── scaler.pkl                    # Data scaler object
├── requirements.txt              # Project dependencies
├── index.html                    # Frontend template
├── precription.txt               # Notes file
└── README.md                     # Documentation
```

## 🚀 Features  
- Preprocessing of Fitbit activity and health data  
- Database integration with **SQLite**  
- Machine Learning pipeline for predictive analysis (Random Forest model included)  
- Data visualization via frontend/web app  
- Evaluation metrics and model testing  

## 🛠️ Technologies Used  
- **Python** (Flask / Streamlit)  
- **Pandas, NumPy, Scikit-learn** for ML  
- **SQLite3** for database  
- **Pickle** for model storage  
- **HTML/CSS** for frontend  

## ⚡ Installation & Setup  
1. Clone the repository  
```bash
git clone https://github.com/KailashSatkuri-warangal/KailashSatkuri-warangal.git
cd KailashSatkuri-warangal
```

2. Create a virtual environment & activate it  
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies  
```bash
pip install -r requirements.txt
```

4. Run the application  
```bash
python app.py
```

The app will start locally on `http://127.0.0.1:5000/`  

## 📊 Dataset  
The project uses Fitbit data collected from **March 2016 to May 2016** across multiple participants. The dataset includes:  
- Steps, Calories, Heart rate  
- Sleep data  
- Activity levels  

## 📈 Machine Learning Model  
- **Random Forest Classifier** trained on preprocessed Fitbit dataset  
- Includes **scaler.pkl** for normalization  
- Evaluation results can be reproduced with `eval.py`  

## 🔮 Future Improvements  
- Integration with **live Fitbit API**  
- More advanced ML/DL models (XGBoost, LSTMs)  
- Interactive dashboards for visualization  

## 👥 Contributors  
- Kailash Satkuri
- Dheeraj Mitta
- Akhilanandateja Sanga
- Rohith Macharla
- Venkata Shiva Sri Chodisetty
## 📌 Live Demo  
[![Vercel App](https://img.shields.io/badge/Vercel-Live%20Demo-black?logo=vercel&style=flat-square)](https://smart-wearable-personal-monitoring-systems.vercel.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reviewvibe.streamlit.app)

