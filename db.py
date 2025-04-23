import sqlite3
conn = sqlite3.connect('health_data.db')
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS health_data")
c.execute('''
CREATE TABLE health_data (
    id INTEGER,
    timestamp TEXT,
    steps REAL,
    distance REAL,
    calories REAL,
    weight REAL,
    bmi REAL,
    heart_rate REAL,
    mobile_usage REAL,
    water_intake REAL,
    systolic_bp REAL,
    diastolic_bp REAL
)
''')
conn.commit()
conn.close()
