@dashboard_bp.route('/stream/<int:machine_id>')
@login_required
def stream_page(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    if machine.user_id != current_user.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('dashboard.index'))
    return render_template('stream.html', machine=machine, features=machine.feature_names)

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
            time.sleep(0.1)  # Small delay for smooth rendering
        
        # Simulate real-time new data (in production, this would read from sensors)
        import random
        counter = 0
        
        while True:
            # Generate new data point (simulate sensor reading)
            new_values = {}
            for feature in machine.feature_names:
                # Add some random variation to last value or use base
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
            
            counter += 1
            time.sleep(2)  # New data every 2 seconds
    
    return Response(event_stream(), mimetype="text/event-stream")

def generate_solution(features, values):
    """Generate solution based on which features are anomalous"""
    solutions = []
    
    # Simple rule-based solutions (can be made more sophisticated)
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