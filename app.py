from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3
from db_config import db_config
import datetime
import re
import joblib           # For AI Risk Model
import numpy as np      # For Data Handling
import os               # For File Paths
import platform         # To detect Windows vs Linux
import pytesseract      # For OCR (Vision)
from PIL import Image   # For Image Processing
import google.generativeai as genai  # For Interactive Chat

app = Flask(__name__)

# --- A. GEMINI API SETUP ---
# Secured via Environment Variables for GitHub safety
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables.")
    model = None

# --- B. OCR / VISION SETUP ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# --- C. RISK MODEL SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'risk_model.pkl')

try:
    risk_model = joblib.load(MODEL_PATH)
    print(f"✅ SUCCESS: AI Risk Model loaded from {MODEL_PATH}")
except Exception as e:
    risk_model = None
    print(f"⚠️ WARNING: Could not load AI model. Error: {e}")

DOCTORS_NAMES = [
    "Dr. A. Smith (Cardiology)", "Dr. B. Jones (Neurology)", "Dr. C. Williams (Orthopedics)",
    "Dr. D. Brown (Pediatrics)", "Dr. E. Davis (General Surgeon)", "Dr. F. Miller (ENT)",
    "Dr. G. Wilson (Dermatology)", "Dr. H. Moore (Gynecology)", "Dr. I. Taylor (Oncology)",
    "Dr. J. Anderson (Psychiatry)"
]

# --- D. DATABASE UTILITIES ---
def get_db_connection():
    try:
        conn = sqlite3.connect(db_config['database'])
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as err:
        print(f"❌ Error: {err}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS hospital (
            Reference_No TEXT PRIMARY KEY, Nameoftablets TEXT, dose TEXT, 
            Numbersoftablets TEXT, lot TEXT, issuedate TEXT, expdate TEXT, 
            dailydose TEXT, storage TEXT, nhsnumber TEXT, patientname TEXT, 
            DOB TEXT, patientaddress TEXT, doctor TEXT, Disease TEXT)''')
        conn.commit()
        conn.close()

def calculate_age(dob_str):
    try:
        for fmt in ["%d-%m-%Y", "%d-%m-%y", "%d/%m/%Y", "%Y-%m-%d"]:
            try:
                birth_date = datetime.datetime.strptime(str(dob_str), fmt).date()
                today = datetime.date.today()
                return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            except: continue
        return 30
    except: return 30

# --- E. OFFLINE HEALTH ADVICE LOGIC ---
def get_health_advice(row):
    advice_list = []
    tablet = row.get('Nameoftablets', '').lower() if row.get('Nameoftablets') else ''
    daily_dose_str = row.get('dailydose', '0')
    dob = row.get('DOB', '')
    storage = row.get('storage', '')
    expdate = row.get('expdate', '')
    disease = row.get('Disease', '')

    exp_date_obj = None
    for fmt in ["%d-%m-%Y", "%d-%m-%y", "%d/%m/%Y", "%Y-%m-%d"]:
        try:
            exp_date_obj = datetime.datetime.strptime(str(expdate), fmt).date()
            break
        except: pass
    
    if exp_date_obj:
        days_left = (exp_date_obj - datetime.date.today()).days
        if days_left < 0: 
            advice_list.append("🔴 <b>CRITICAL:</b> Medicine EXPIRED! Do not consume.")
        elif days_left < 30: 
            advice_list.append(f"⚠️ <b>Expiry Warning:</b> Expires in {days_left} days.")

    if "corona" in tablet or "vaccine" in tablet: 
        advice_list.append("💉 <b>Vaccine Care:</b> Mild fever is normal. Rest for 2 days.")
    elif "acetaminophen" in tablet: 
        advice_list.append("💊 <b>Pain/Fever:</b> Take after food. Do not exceed dose.")
    elif "paracetamol" in tablet or "dollo" in tablet: 
        advice_list.append("🌡️ <b>Fever:</b> Monitor temperature. Gap of 6 hours between doses.")
    elif "ativan" in tablet: 
        advice_list.append("💤 <b>Anxiety/Sleep:</b> May cause drowsiness. Do not drive.")
    else: 
        advice_list.append("ℹ️ <b>General:</b> Complete the full course as prescribed.")

    age = calculate_age(dob)
    if age > 60: advice_list.append("👴 <b>Senior Care:</b> Drink water frequently, watch for dizziness.")
    if age < 12: advice_list.append("👶 <b>Child Care:</b> Ensure dosage is strictly by weight.")

    if risk_model:
        try:
            try: dose_val = int(re.search(r'\d+', str(daily_dose_str)).group())
            except: dose_val = 1
            is_anti = 1 if any(x in tablet for x in ['cillin', 'mycin', 'oxacin']) else 0
            is_pain = 1 if any(x in tablet for x in ['pain', 'acetaminophen', 'dol']) else 0
            prediction = risk_model.predict([[age, dose_val, is_anti, is_pain]])
            if prediction[0] == 1: advice_list.append("🤖 <b>AI RISK ALERT:</b> High-risk dosage pattern detected.")
            else: advice_list.append("🤖 <b>AI Analysis:</b> Dosage looks standard.")
        except: pass

    if storage and "fridge" in storage.lower(): advice_list.append("❄️ <b>Storage:</b> Keep Refrigerated.")
    if disease: advice_list.append(f"🩺 <b>Condition Note:</b> Managing {disease}.")
    return advice_list

# --- F. ROUTES ---

@app.route('/')
def index():
    conn = get_db_connection()
    if conn:
        rows = conn.execute("SELECT * FROM hospital").fetchall()
        patients = [dict(row) for row in rows]
        conn.close()
        return render_template('index.html', patients=patients)
    return "Database Connection Failed."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        req = request.json
        user_text = req.get('message', '').strip()
        
        # Extract Reference Number if present in message or context
        match = re.search(r"(REF\d+|ref\d+|\d{4})", user_text)
        ref_to_use = match.group(0).upper() if match else req.get('context_ref', '')

        patient_context = "No specific patient context provided."
        tips_html = ""

        if ref_to_use:
            conn = get_db_connection()
            db_row = conn.execute("SELECT * FROM hospital WHERE Reference_No=?", (ref_to_use,)).fetchone()
            conn.close()

            if db_row:
                row = dict(db_row)
                # Generate the offline medical advice tips
                tips_list = get_health_advice(row)
                tips_html = "<br>".join(tips_list)
                patient_context = (f"Patient: {row['patientname']}, Age: {calculate_age(row['DOB'])}, "
                                   f"Disease: {row.get('Disease', 'Unknown')}, Medicine: {row['Nameoftablets']}, "
                                   f"RefID: {row['Reference_No']}")

        # Instruction to Gemini: Handle medical AND general queries
        prompt = (f"System Context (Patient Data): {patient_context}\n"
                  f"User Query: {user_text}\n\n"
                  f"You are a helpful AI Medical Assistant for a Hospital Management System. "
                  f"1. If the user asks about the patient above, use the System Context to answer. "
                  f"2. If the user asks a general question (e.g., world news, history, or greetings), answer politely and accurately. "
                  f"Keep your answers brief and professional.")

        if model:
            try:
                ai_response = model.generate_content(prompt)
                bot_reply = ai_response.text.replace("\n", "<br>")
                
                # Combine Gemini's answer with our Offline Safety Tips (if a patient was found)
                final_response = f"🤖 <b>AI:</b> {bot_reply}"
                if tips_html:
                    final_response += f"<br><br><b>💡 Health & Safety Tips (Offline):</b><br>{tips_html}"
                
                return jsonify({"response": final_response, "ref": ref_to_use})
            except Exception as e:
                # If Gemini fails (offline/quota), provide the Offline Tips only
                if tips_html:
                    return jsonify({"response": f"<b>💡 Offline Advice for {ref_to_use}:</b><br>{tips_html}", "ref": ref_to_use})
                return jsonify({"response": "I am having trouble connecting to my AI brain. Please check your internet."})

        return jsonify({"response": "AI Model not loaded correctly. Please check GEMINI_API_KEY."})

    except Exception as e:
        return jsonify({"response": f"Error: {e}"}), 500

@app.route('/add', methods=['POST'])
def add_patient():
    data = request.form
    conn = get_db_connection()
    if conn:
        sql = """INSERT INTO hospital (Nameoftablets, Reference_No, dose, Numbersoftablets, lot, issuedate, expdate, dailydose, storage, nhsnumber, patientname, DOB, patientaddress, doctor, Disease) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        val = (data['name'], data['ref'], data['dose'], data['no_of_tablets'], data['lot'], data['issue_date'], 
               data['exp_date'], data['daily_dose'], data['storage'], data['nhs'], data['pname'], data['dob'], 
               data['address'], data['doctor'], data.get('disease', ''))
        conn.execute(sql, val)
        conn.commit()
        conn.close()
    return redirect(url_for('index'))

@app.route('/delete/<ref>', methods=['GET'])
def delete_patient(ref):
    conn = get_db_connection()
    if conn:
        conn.execute("DELETE FROM hospital WHERE Reference_No=?", (ref,))
        conn.commit()
        conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)