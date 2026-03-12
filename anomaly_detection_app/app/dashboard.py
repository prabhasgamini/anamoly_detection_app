from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, Response, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models import Machine, DataPoint, Alert
from app.utils.data_parser import parse_csv
from app.utils.email_alert import send_alert
from app.anomaly.detector import train_models_for_machine, load_detector_for_machine
import os
import json
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime

# Create the blueprint FIRST
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    """Dashboard home page showing all machines"""
    machines = Machine.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', machines=machines)

@dashboard_bp.route('/configure', methods=['GET', 'POST'])
@login_required
def configure():
    if request.method == 'POST':
        machine_id = request.form['machine_id']
        feature_count = int(request.form['feature_count'])
        feature_names = []
        for i in range(feature_count):
            name = request.form.get(f'feature_name_{i}')
            if name:
                feature_names.append(name)
        
        machine = Machine(
            machine_id=machine_id,
            user_id=current_user.id,
            feature_count=feature_count,
            feature_names=feature_names
        )
        db.session.add(machine)
        db.session.commit()
        flash('Machine configured. Now upload training data.', 'success')
        return redirect(url_for('dashboard.upload', machine_id=machine.id))
    
    return render_template('configure_features.html')

@dashboard_bp.route('/upload/<int:machine_id>', methods=['GET', 'POST'])
@login_required
def upload(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    if machine.user_id != current_user.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('dashboard.index'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(f"user_{current_user.id}_machine_{machine_id}.csv")
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Parse CSV
            try:
                df = parse_csv(filepath, machine.feature_names)
            except Exception as e:
                flash(f'Error parsing CSV: {str(e)}', 'danger')
                return redirect(request.url)

            # Train models
            try:
                train_models_for_machine(machine, df)
                flash('Models trained successfully!', 'success')
            except Exception as e:
                flash(f'Model training failed: {str(e)}', 'danger')
                return redirect(request.url)

            # Save data points
            for _, row in df.iterrows():
                values = {feature: float(row[feature]) for feature in machine.feature_names}
                dp = DataPoint(
                    machine_id=machine.id,
                    timestamp=row['timestamp'],
                    values=values,
                    is_anomaly=False
                )
                db.session.add(dp)
            db.session.commit()

            return redirect(url_for('dashboard.index'))
        else:
            flash('Please upload a CSV file', 'danger')
    
    return render_template('upload.html', machine=machine)

@dashboard_bp.route('/stream/<int:machine_id>')
@login_required
def stream_page(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    if machine.user_id != current_user.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('dashboard.index'))
    return render_template('stream.html', machine=machine, features=machine.feature_names)

@dashboard_bp.route('/visualize/<int:machine_id>')
@login_required
def visualize(machine_id):
    """Page for feature-wise visualizations"""
    machine = Machine.query.get_or_404(machine_id)
    if machine.user_id != current_user.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('dashboard.index'))
    
    # Get all data points for this machine
    data_points = DataPoint.query.filter_by(machine_id=machine_id)\
        .order_by(DataPoint.timestamp)\
        .all()
    
    # Prepare data for template
    chart_data = {
        'timestamps': [],
        'features': {}
    }
    
    # Initialize feature data structures
    for feature in machine.feature_names:
        chart_data['features'][feature] = {
            'values': [],
            'anomalies': []
        }
    
    # Populate data
    for dp in data_points:
        # Format timestamp safely
        if dp.timestamp:
            try:
                chart_data['timestamps'].append(dp.timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            except:
                chart_data['timestamps'].append(str(dp.timestamp))
        else:
            chart_data['timestamps'].append('Unknown')
        
        # Get values for each feature
        for feature in machine.feature_names:
            try:
                value = float(dp.values.get(feature, 0)) if dp.values else 0
            except (TypeError, ValueError):
                value = 0
            chart_data['features'][feature]['values'].append(value)
            
            # Get anomaly flag
            anomaly = 1 if dp.is_anomaly else 0
            chart_data['features'][feature]['anomalies'].append(anomaly)
    
    return render_template('visualize.html', 
                         machine=machine, 
                         chart_data=chart_data,
                         features=machine.feature_names,
                         has_data=len(data_points) > 0)

@dashboard_bp.route('/api/stream/<int:machine_id>')
@login_required
def stream_data(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    if machine.user_id != current_user.id:
        return "Unauthorized", 403

    def event_stream():
        # Load models and pure pattern detector
        iso_model, lstm_detector, hybrid = load_detector_for_machine(machine)
        
        # Get existing data points for initial plot
        data_points = DataPoint.query.filter_by(machine_id=machine_id)\
            .order_by(DataPoint.timestamp)\
            .all()
        
        # Send historical data first
        for dp in data_points:
            data = {
                'type': 'historical',
                'timestamp': dp.timestamp.isoformat() if dp.timestamp else str(dp.timestamp),
                'values': dp.values,
                'anomaly': dp.is_anomaly,
                'features': machine.feature_names
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)
        
        # Simulate real-time new data
        import random
        
        while True:
            # Generate new data point
            new_values = {}
            for feature in machine.feature_names:
                if data_points:
                    last_val = data_points[-1].values.get(feature, 50)
                    new_values[feature] = last_val + random.gauss(0, 2)
                else:
                    new_values[feature] = 50 + random.gauss(0, 5)
            
            # Create timestamp
            new_timestamp = datetime.now()
            
            # Create row for detection
            row_dict = {'timestamp': new_timestamp}
            row_dict.update(new_values)
            row_series = pd.Series(row_dict)
            
            # Detect anomaly using pure pattern learning
            from app.anomaly.isolation_forest_model import score_isolation_forest

            iso_raw = score_isolation_forest(iso_model, row_series[machine.feature_names].values)
            lstm_error = lstm_detector.update(row_series)

            is_anomaly, confidence, reason = hybrid.detect(iso_raw, lstm_error)
            
            # Save to database
            dp = DataPoint(
                machine_id=machine.id,
                timestamp=new_timestamp,
                values=new_values,
                is_anomaly=is_anomaly
            )
            db.session.add(dp)
            db.session.commit()
            
            # Send alert if anomaly
            if is_anomaly:
                solution = generate_solution(machine.feature_names, new_values)
                alert_msg = f"Anomaly detected at {new_timestamp.strftime('%H:%M:%S')}\nReason: {reason}\n{solution}"
                
                # Send email alert
                threading.Thread(target=send_alert, args=(
                    current_user.email,
                    machine.machine_id,
                    new_timestamp.isoformat(),
                    alert_msg
                )).start()
                
                # Save alert
                alert = Alert(
                    machine_id=machine.id,
                    data_point_id=dp.id,
                    message=alert_msg
                )
                db.session.add(alert)
                db.session.commit()
            
            # Send to frontend
            data = {
                'type': 'realtime',
                'timestamp': new_timestamp.isoformat(),
                'values': new_values,
                'anomaly': is_anomaly,
                'confidence': float(confidence),
                'reason': reason,
                'features': machine.feature_names
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            time.sleep(2)
    
    return Response(event_stream(), mimetype="text/event-stream")

@dashboard_bp.route('/api/data/<int:machine_id>')
@login_required
def get_machine_data(machine_id):
    """API endpoint to get machine data for charts"""
    machine = Machine.query.get_or_404(machine_id)
    if machine.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data_points = DataPoint.query.filter_by(machine_id=machine_id)\
        .order_by(DataPoint.timestamp.desc())\
        .limit(100)\
        .all()
    
    result = {
        'timestamps': [],
        'features': {name: [] for name in machine.feature_names},
        'anomalies': []
    }
    
    for dp in reversed(data_points):
        if dp.timestamp:
            result['timestamps'].append(dp.timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            result['timestamps'].append('Unknown')
            
        for name in machine.feature_names:
            result['features'][name].append(dp.values.get(name, 0))
        result['anomalies'].append(1 if dp.is_anomaly else 0)
    
    return jsonify(result)

def generate_solution(features, values):
    """Generate solution based on which features are anomalous"""
    solutions = []
    
    for feature, value in values.items():
        if "temperature" in feature.lower() and value > 80:
            solutions.append("🌡️ High temperature: Check cooling system")
        elif "vibration" in feature.lower() and value > 0.8:
            solutions.append("⚡ High vibration: Check bearing alignment")
        elif "current" in feature.lower() and value > 12:
            solutions.append("🔌 High current: Check for overload")
        elif "pressure" in feature.lower() and value > 110:
            solutions.append("💨 High pressure: Check valves and pipes")
        elif "rpm" in feature.lower() and (value < 1400 or value > 1600):
            solutions.append("⚙️ RPM deviation: Check motor speed controller")
    
    if not solutions:
        solutions.append("🔧 General maintenance check recommended")
    
    return " | ".join(solutions)