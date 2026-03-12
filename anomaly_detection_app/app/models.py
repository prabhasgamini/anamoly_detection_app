from app import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    machines = db.relationship('Machine', backref='owner', lazy=True)

class Machine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.String(50), unique=True, nullable=False)  # user-provided ID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    feature_count = db.Column(db.Integer, nullable=False)
    feature_names = db.Column(db.JSON, nullable=False)  # list of names
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    iso_model_path = db.Column(db.String(200))
    lstm_model_path = db.Column(db.String(200))
    iso_threshold = db.Column(db.Float)
    iso_min_score = db.Column(db.Float)
    lstm_threshold = db.Column(db.Float)
    lstm_max_error = db.Column(db.Float)

class DataPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    values = db.Column(db.JSON, nullable=False)  # dict {feature_name: value}
    is_anomaly = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    data_point_id = db.Column(db.Integer, db.ForeignKey('data_point.id'))
    message = db.Column(db.Text)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
