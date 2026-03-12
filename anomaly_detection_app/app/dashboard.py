import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models import Machine, DataPoint, Alert
from app.utils.data_parser import parse_csv, parse_timestamp
from app.utils.email_alert import send_alert
from app.anomaly.detector import train_models_for_machine, load_detector_for_machine
import time
import threading

dashboard_bp = Blueprint('dashboard', __name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@dashboard_bp.route('/')
@login_required
def index():
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
            filepath = os.path.join(UPLOAD_FOLDER, filename)
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

            # Optionally store data points
            for _, row in df.iterrows():
                dp = DataPoint(
                    machine_id=machine.id,
                    timestamp=row['timestamp'],
                    values={col: row[col] for col in machine.feature_names},
                    is_anomaly=False  # will be set during streaming
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
    return render_template('stream.html', machine=machine)

# SSE endpoint for streaming
@dashboard_bp.route('/api/stream/<int:machine_id>')
@login_required
def stream_data(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    if machine.user_id != current_user.id:
        return "Unauthorized", 403

    def event_stream():
        # Load models and detectors
        iso_model, lstm_detector, hybrid = load_detector_for_machine(machine)

        # Simulate streaming from the stored data points (or live sensor)
        # For demo, we loop through existing data points
        data_points = DataPoint.query.filter_by(machine_id=machine_id).order_by(DataPoint.timestamp).all()
        for dp in data_points:
            # Convert to row dict
            row_dict = {'timestamp': dp.timestamp.isoformat()}
            row_dict.update(dp.values)

            # Get raw scores
            # Build a pandas Series for compatibility
            import pandas as pd
            row_series = pd.Series(row_dict)

            iso_raw = iso_model.decision_function(row_series[machine.feature_names].values.reshape(1, -1))[0]
            lstm_error = lstm_detector.update(row_series)

            # Combine
            combined_score, anomaly_flag = hybrid.combine(iso_raw, lstm_error)

            # If anomaly, send alert and save to DB
            if anomaly_flag:
                # Send email (in background to not block)
                threading.Thread(target=send_alert, args=(
                    current_user.email,
                    machine.machine_id,
                    dp.timestamp.isoformat(),
                    f"Anomaly detected with score {combined_score:.2f}"
                )).start()
                # Save alert record
                alert = Alert(machine_id=machine.id, data_point_id=dp.id, message=f"Score: {combined_score:.2f}")
                db.session.add(alert)
                db.session.commit()

            # Yield event
            data = {
                'timestamp': dp.timestamp.isoformat(),
                'values': dp.values,
                'anomaly': bool(anomaly_flag),
                'score': combined_score
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)  # simulate real-time interval

    return Response(event_stream(), mimetype="text/event-stream")
