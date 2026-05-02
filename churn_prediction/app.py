from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import pickle, json, os, hashlib, uuid
from datetime import datetime
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'churnbank_secret_2024'

# ── Load models ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
def load(name): return pickle.load(open(os.path.join(BASE,'models',name),'rb'))

rf      = load('random_forest.pkl')
gb      = load('gradient_boost.pkl')
lr      = load('logistic_reg.pkl')
scaler  = load('scaler.pkl')
le_geo  = load('le_geography.pkl')
le_gen  = load('le_gender.pkl')
le_con  = load('le_contract.pkl')
FEATURES= load('feature_names.pkl')

# ── In-memory "database" (Vercel-compatible) ──────────────────────────────────
# NOTE: Data resets on each cold start — acceptable for demo/portfolio use
_DB = {}

def load_db():
    return _DB

def save_db(db):
    _DB.update(db)

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def get_user():
    if 'user_email' not in session: return None
    return _DB.get(session['user_email'])

def save_user(user):
    _DB[user['email']] = user

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_email' in session: return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        db = load_db()
        email = request.form['email'].lower().strip()
        if email in db:
            return render_template('register.html', error='Email already registered.')
        pw = request.form['password']
        if len(pw) < 6:
            return render_template('register.html', error='Password must be at least 6 characters.')
        if pw != request.form['confirm_password']:
            return render_template('register.html', error='Passwords do not match.')
        user = {
            'id': str(uuid.uuid4()),
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'email': email,
            'phone': request.form['phone'],
            'dob': request.form['dob'],
            'gender': request.form['gender'],
            'geography': request.form['geography'],
            'password': hash_pw(pw),
            'balance': 0.0,
            'total_deposits': 0.0,
            'total_withdrawals': 0.0,
            'credit_score': 650,
            'transactions': [],
            'predictions': [],
            'loans': [],
            'fds': [],
            'acc_no': str(np.random.randint(100000000,999999999)),
            'joined': datetime.now().strftime('%d %b %Y')
        }
        db[email] = user
        save_db(db)
        flash('Account created! Please sign in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        db = load_db()
        email = request.form['email'].lower().strip()
        pw    = request.form['password']
        user  = db.get(email)
        if not user or user['password'] != hash_pw(pw):
            return render_template('login.html', error='Invalid email or password.')
        session['user_email'] = email
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    user = get_user()
    if not user: return redirect(url_for('login'))
    recent = user['transactions'][-5:][::-1]
    last_pred = user['predictions'][-1] if user['predictions'] else None
    return render_template('dashboard.html', user=user, recent=recent, last_pred=last_pred)

# ── Banking Operations ────────────────────────────────────────────────────────
@app.route('/deposit', methods=['GET','POST'])
def deposit():
    user = get_user()
    if not user: return redirect(url_for('login'))
    if request.method == 'POST':
        amt = float(request.form['amount'])
        if amt <= 0: return render_template('deposit.html', user=user, error='Enter a valid amount.')
        user['balance'] += amt
        user['total_deposits'] += amt
        user['transactions'].append({
            'id': 'TXN'+str(uuid.uuid4())[:8].upper(),
            'date': datetime.now().strftime('%d %b %Y %H:%M'),
            'type': 'Credit', 'category': 'Deposit',
            'description': f"Deposit via {request.form['source']}",
            'note': request.form.get('note',''),
            'amount': amt
        })
        save_user(user)
        flash(f'Successfully deposited ₹{amt:,.2f}!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('deposit.html', user=user)

@app.route('/withdraw', methods=['GET','POST'])
def withdraw():
    user = get_user()
    if not user: return redirect(url_for('login'))
    if request.method == 'POST':
        amt = float(request.form['amount'])
        if amt <= 0: return render_template('withdraw.html', user=user, error='Enter a valid amount.')
        if amt > user['balance']: return render_template('withdraw.html', user=user, error=f"Insufficient balance. Available: ₹{user['balance']:,.2f}")
        user['balance'] -= amt
        user['total_withdrawals'] += amt
        user['transactions'].append({
            'id': 'TXN'+str(uuid.uuid4())[:8].upper(),
            'date': datetime.now().strftime('%d %b %Y %H:%M'),
            'type': 'Debit', 'category': 'Withdrawal',
            'description': f"Withdrawal via {request.form['method']}",
            'note': request.form.get('note',''),
            'amount': amt
        })
        save_user(user)
        flash(f'Successfully withdrawn ₹{amt:,.2f}!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('withdraw.html', user=user)

@app.route('/transfer', methods=['GET','POST'])
def transfer():
    user = get_user()
    if not user: return redirect(url_for('login'))
    if request.method == 'POST':
        amt = float(request.form['amount'])
        if amt <= 0: return render_template('transfer.html', user=user, error='Enter a valid amount.')
        if amt > user['balance']: return render_template('transfer.html', user=user, error='Insufficient balance.')
        user['balance'] -= amt
        user['total_withdrawals'] += amt
        name = request.form['recipient_name']
        acc  = request.form['recipient_acc']
        ttype= request.form['transfer_type']
        user['transactions'].append({
            'id': 'TXN'+str(uuid.uuid4())[:8].upper(),
            'date': datetime.now().strftime('%d %b %Y %H:%M'),
            'type': 'Debit', 'category': 'Transfer',
            'description': f"{ttype} to {name} ({acc})",
            'note': request.form.get('remarks',''),
            'amount': amt
        })
        save_user(user)
        flash(f'Successfully transferred ₹{amt:,.2f} to {name}!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('transfer.html', user=user)

@app.route('/loan', methods=['GET','POST'])
def loan():
    user = get_user()
    if not user: return redirect(url_for('login'))
    if request.method == 'POST':
        amt    = float(request.form['amount'])
        tenure = int(request.form['tenure'])
        ltype  = request.form['loan_type']
        rates  = {'Personal Loan':12,'Home Loan':8.5,'Car Loan':9.0,'Education Loan':8.0,'Business Loan':11.0}
        rate   = rates.get(ltype, 12)
        mr     = rate/12/100
        emi    = amt*mr*(1+mr)**tenure/((1+mr)**tenure-1)
        user['balance'] += amt
        user['total_deposits'] += amt
        user['loans'].append({
            'id': 'LN'+str(uuid.uuid4())[:6].upper(),
            'type': ltype, 'amount': amt, 'tenure': tenure,
            'rate': rate, 'emi': round(emi,2),
            'purpose': request.form.get('purpose',''),
            'date': datetime.now().strftime('%d %b %Y'), 'status': 'Active'
        })
        user['transactions'].append({
            'id': 'TXN'+str(uuid.uuid4())[:8].upper(),
            'date': datetime.now().strftime('%d %b %Y %H:%M'),
            'type': 'Credit', 'category': 'Loan',
            'description': f"{ltype} disbursed",
            'note': f"EMI: ₹{emi:,.2f}/mo for {tenure} months @ {rate}% p.a.",
            'amount': amt
        })
        save_user(user)
        flash(f'{ltype} of ₹{amt:,.2f} approved! Monthly EMI: ₹{emi:,.2f}', 'success')
        return redirect(url_for('dashboard'))
    return render_template('loan.html', user=user)

@app.route('/fixed_deposit', methods=['GET','POST'])
def fixed_deposit():
    user = get_user()
    if not user: return redirect(url_for('login'))
    if request.method == 'POST':
        amt    = float(request.form['amount'])
        tenure_rate = request.form['tenure_rate'].split('|')
        months = int(tenure_rate[0]); rate = float(tenure_rate[1])
        if amt < 1000: return render_template('fixed_deposit.html', user=user, error='Minimum FD amount is ₹1,000.')
        if amt > user['balance']: return render_template('fixed_deposit.html', user=user, error='Insufficient balance.')
        maturity = amt * (1 + rate/100 * months/12)
        user['balance'] -= amt
        user['total_withdrawals'] += amt
        user['fds'].append({
            'id': 'FD'+str(uuid.uuid4())[:6].upper(),
            'amount': amt, 'months': months, 'rate': rate,
            'maturity': round(maturity,2),
            'nominee': request.form.get('nominee',''),
            'date': datetime.now().strftime('%d %b %Y'), 'status': 'Active'
        })
        user['transactions'].append({
            'id': 'TXN'+str(uuid.uuid4())[:8].upper(),
            'date': datetime.now().strftime('%d %b %Y %H:%M'),
            'type': 'Debit', 'category': 'Fixed Deposit',
            'description': f"FD created for {months} months @ {rate}% p.a.",
            'note': f"Maturity: ₹{maturity:,.2f}",
            'amount': amt
        })
        save_user(user)
        flash(f'FD created! Maturity value: ₹{maturity:,.2f}', 'success')
        return redirect(url_for('dashboard'))
    return render_template('fixed_deposit.html', user=user)

@app.route('/pay_bills', methods=['GET','POST'])
def pay_bills():
    user = get_user()
    if not user: return redirect(url_for('login'))
    if request.method == 'POST':
        amt   = float(request.form['amount'])
        btype = request.form['bill_type']
        prov  = request.form['provider']
        cons  = request.form['consumer_no']
        if amt > user['balance']: return render_template('pay_bills.html', user=user, error='Insufficient balance.')
        user['balance'] -= amt
        user['total_withdrawals'] += amt
        user['transactions'].append({
            'id': 'TXN'+str(uuid.uuid4())[:8].upper(),
            'date': datetime.now().strftime('%d %b %Y %H:%M'),
            'type': 'Debit', 'category': 'Bill Payment',
            'description': f"{btype} — {prov}",
            'note': f"Consumer: {cons}",
            'amount': amt
        })
        save_user(user)
        flash(f'{btype} bill of ₹{amt:,.2f} paid to {prov}!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('pay_bills.html', user=user)

@app.route('/history')
def history():
    user = get_user()
    if not user: return redirect(url_for('login'))
    txns = user['transactions'][::-1]
    category = request.args.get('category','all')
    if category != 'all':
        txns = [t for t in txns if t['category'] == category]
    return render_template('history.html', user=user, txns=txns, category=category)

# ── Churn Prediction ──────────────────────────────────────────────────────────
@app.route('/churn', methods=['GET','POST'])
def churn():
    user = get_user()
    if not user: return redirect(url_for('login'))

    months_active = max(1, len(user['transactions']) // 2 + 1)
    avg_charges   = round(user['total_withdrawals'] / max(1, months_active), 2)
    n_products    = 1 + (1 if user['loans'] else 0) + (1 if user['fds'] else 0)
    has_loan      = 1 if user['loans'] else 0
    has_fd        = 1 if user['fds'] else 0

    prefill = {
        'tenure': months_active,
        'age': 30,
        'balance': round(user['balance'], 2),
        'salary': 50000,
        'credit_score': user.get('credit_score', 650),
        'products': n_products,
        'has_card': 1,
        'is_active': 1,
        'monthly_charges': avg_charges or 500,
        'complaints': 0,
        'geography': user.get('geography','Hyderabad'),
        'gender': user.get('gender','Male'),
        'contract': 'Month-to-Month',
        'loan': has_loan,
        'fd': has_fd,
    }

    result = None
    if request.method == 'POST':
        try:
            geo_enc = le_geo.transform([request.form['geography']])[0]
            gen_enc = le_gen.transform([request.form['gender']])[0]
            con_enc = le_con.transform([request.form['contract']])[0]

            row = np.array([[
                float(request.form['tenure']),
                float(request.form['age']),
                float(request.form['balance']),
                float(request.form['salary']),
                float(request.form['credit_score']),
                int(request.form['products']),
                int(request.form['has_card']),
                int(request.form['is_active']),
                float(request.form['monthly_charges']),
                int(request.form['complaints']),
                geo_enc, gen_enc, con_enc,
                int(request.form['loan']),
                int(request.form['fd']),
            ]])

            row_scaled = scaler.transform(row)

            rf_prob = rf.predict_proba(row)[0][1]
            gb_prob = gb.predict_proba(row)[0][1]
            lr_prob = lr.predict_proba(row_scaled)[0][1]
            ensemble_prob = round((rf_prob*0.4 + gb_prob*0.4 + lr_prob*0.2), 4)

            risk = 'High' if ensemble_prob >= 0.65 else 'Medium' if ensemble_prob >= 0.35 else 'Low'
            score = int(ensemble_prob * 100)

            importances = dict(zip(FEATURES, rf.feature_importances_))
            top_factors = sorted(importances.items(), key=lambda x: -x[1])[:5]

            recommendations = []
            f = {k: float(v) for k,v in request.form.items() if k not in ['geography','gender','contract','csrf_token']}
            if f.get('tenure',0) < 12:   recommendations.append("Customer is new — offer welcome bonus or loyalty rewards.")
            if f.get('monthly_charges',0) > 3000: recommendations.append("High charges — consider offering a discounted plan.")
            if f.get('complaints',0) >= 2: recommendations.append("Multiple complaints — assign dedicated relationship manager.")
            if not f.get('loan',0):       recommendations.append("No loan — offer pre-approved personal loan.")
            if not f.get('fd',0):         recommendations.append("No FD — promote fixed deposit with higher interest rate.")
            if f.get('balance',0) < 5000: recommendations.append("Low balance — offer zero-balance or cashback account upgrade.")
            if not recommendations: recommendations.append("Customer looks stable — maintain regular engagement and rewards.")

            pred_record = {
                'id': 'PRED'+str(uuid.uuid4())[:6].upper(),
                'date': datetime.now().strftime('%d %b %Y %H:%M'),
                'probability': ensemble_prob,
                'risk_level': risk,
                'score': score,
                'rf_prob': round(rf_prob,4),
                'gb_prob': round(gb_prob,4),
                'lr_prob': round(lr_prob,4),
                'input': dict(request.form)
            }
            user['predictions'].append(pred_record)
            save_user(user)

            result = {
                'probability': ensemble_prob,
                'percent': score,
                'risk': risk,
                'rf': round(rf_prob*100,1),
                'gb': round(gb_prob*100,1),
                'lr': round(lr_prob*100,1),
                'top_factors': top_factors,
                'recommendations': recommendations,
                'pred_id': pred_record['id']
            }
            prefill = dict(request.form)
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')

    history_preds = user['predictions'][::-1][:10]
    return render_template('churn.html', user=user, prefill=prefill, result=result, history=history_preds)

@app.route('/profile', methods=['GET','POST'])
def profile():
    user = get_user()
    if not user: return redirect(url_for('login'))
    if request.method == 'POST':
        if request.form.get('action') == 'change_password':
            if user['password'] != hash_pw(request.form['old_password']):
                flash('Current password is incorrect.', 'danger')
            elif request.form['new_password'] != request.form['confirm_password']:
                flash('New passwords do not match.', 'danger')
            elif len(request.form['new_password']) < 6:
                flash('Password must be at least 6 characters.', 'danger')
            else:
                user['password'] = hash_pw(request.form['new_password'])
                save_user(user)
                flash('Password updated successfully!', 'success')
    return render_template('profile.html', user=user)

@app.route('/api/stats')
def api_stats():
    user = get_user()
    if not user: return jsonify({'error': 'Unauthorized'}), 401
    preds = user.get('predictions', [])
    return jsonify({
        'balance': user['balance'],
        'total_deposits': user['total_deposits'],
        'total_withdrawals': user['total_withdrawals'],
        'transactions': len(user['transactions']),
        'predictions': len(preds),
        'last_risk': preds[-1]['risk_level'] if preds else None,
        'last_prob': preds[-1]['probability'] if preds else None,
    })

# ── Bulk Prediction ───────────────────────────────────────────────────────────
import io, csv
from flask import Response

bulk_results_store = {}

@app.route('/bulk_churn', methods=['GET','POST'])
def bulk_churn():
    user = get_user()
    if not user: return redirect(url_for('login'))
    results = None
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or file.filename == '':
            flash('Please upload a CSV file.', 'danger')
            return render_template('bulk_churn.html', user=user, results=None)
        try:
            stream = io.StringIO(file.stream.read().decode('utf-8'))
            reader = csv.DictReader(stream)
            rows = []
            for i, row in enumerate(reader):
                try:
                    geo_enc = le_geo.transform([row['geography']])[0]
                    gen_enc = le_gen.transform([row['gender']])[0]
                    con_enc = le_con.transform([row['contract']])[0]
                    data = np.array([[
                        float(row['tenure']), float(row['age']),
                        float(row['balance']), float(row['salary']),
                        float(row['credit_score']), int(row['products']),
                        int(row['has_card']), int(row['is_active']),
                        float(row['monthly_charges']), int(row['complaints']),
                        geo_enc, gen_enc, con_enc,
                        int(row['loan']), int(row['fd'])
                    ]])
                    data_scaled = scaler.transform(data)
                    rf_p  = rf.predict_proba(data)[0][1]
                    gb_p  = gb.predict_proba(data)[0][1]
                    lr_p  = lr.predict_proba(data_scaled)[0][1]
                    prob  = round(rf_p*0.4 + gb_p*0.4 + lr_p*0.2, 4)
                    risk  = 'High' if prob >= 0.65 else 'Medium' if prob >= 0.35 else 'Low'
                    action = 'Call Immediately' if risk=='High' else 'Send Offer Email' if risk=='Medium' else 'No Action Needed'
                    rows.append({
                        'customer_id': row.get('customer_id', f'CUST{i+1:04d}'),
                        'tenure': row['tenure'], 'age': row['age'],
                        'balance': float(row['balance']),
                        'monthly_charges': float(row['monthly_charges']),
                        'complaints': row['complaints'],
                        'contract': row['contract'],
                        'probability': prob,
                        'percent': int(prob*100),
                        'risk': risk, 'action': action
                    })
                except Exception:
                    continue

            high   = [r for r in rows if r['risk']=='High']
            med    = [r for r in rows if r['risk']=='Medium']
            low    = [r for r in rows if r['risk']=='Low']
            total  = len(rows)

            results = {
                'total': total,
                'high_count': len(high), 'high_pct': round(len(high)/max(1,total)*100,1),
                'med_count':  len(med),  'med_pct':  round(len(med)/max(1,total)*100,1),
                'low_count':  len(low),  'low_pct':  round(len(low)/max(1,total)*100,1),
                'rows': sorted(rows, key=lambda x: -x['probability'])
            }
            bulk_results_store[user['email']] = results
        except Exception as e:
            flash(f'Error reading CSV: {str(e)}', 'danger')

    saved = bulk_results_store.get(user['email']) if not results else None
    return render_template('bulk_churn.html', user=user, results=results or saved)

@app.route('/download_results')
def download_results():
    user = get_user()
    if not user: return redirect(url_for('login'))
    results = bulk_results_store.get(user['email'])
    if not results:
        flash('No results to download. Run a bulk prediction first.', 'danger')
        return redirect(url_for('bulk_churn'))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['customer_id','tenure','age','balance','monthly_charges','complaints','contract','churn_probability','risk_level','action'])
    for r in results['rows']:
        writer.writerow([r['customer_id'],r['tenure'],r['age'],r['balance'],r['monthly_charges'],r['complaints'],r['contract'],f"{r['percent']}%",r['risk'],r['action']])
    output.seek(0)
    return Response(output, mimetype='text/csv', headers={'Content-Disposition':'attachment;filename=churn_predictions.csv'})

@app.route('/download_sample_csv')
def download_sample_csv():
    user = get_user()
    if not user: return redirect(url_for('login'))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['customer_id','tenure','age','balance','salary','credit_score','products','has_card','is_active','monthly_charges','complaints','geography','gender','contract','loan','fd'])
    sample_data = [
        ['C001',3,28,5000,30000,580,1,1,1,2500,2,'Hyderabad','Male','Month-to-Month',0,0],
        ['C002',45,52,120000,90000,780,3,1,1,800,0,'Mumbai','Female','Two Year',1,1],
        ['C003',8,35,2000,25000,620,2,0,0,3500,1,'Delhi','Male','Month-to-Month',0,0],
        ['C004',36,41,75000,60000,720,2,1,1,1200,0,'Bangalore','Female','One Year',1,0],
        ['C005',2,24,500,18000,540,1,0,1,4000,3,'Chennai','Male','Month-to-Month',0,0],
        ['C006',60,55,200000,150000,820,4,1,1,600,0,'Hyderabad','Female','Two Year',1,1],
        ['C007',12,33,8000,40000,650,2,1,0,1800,1,'Mumbai','Male','Month-to-Month',1,0],
        ['C008',24,47,50000,70000,700,3,1,1,900,0,'Delhi','Female','One Year',0,1],
    ]
    for row in sample_data:
        writer.writerow(row)
    output.seek(0)
    return Response(output, mimetype='text/csv', headers={'Content-Disposition':'attachment;filename=sample_customers.csv'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
