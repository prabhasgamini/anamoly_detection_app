from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config
import os
import sys

print("=== Starting Application ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    print("=== Creating Flask App ===")
    app = Flask(__name__)
    print("Loading config...")
    app.config.from_object(Config)

    print("Initializing db...")
    db.init_app(app)
    print("Initializing login manager...")
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    from app.models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    print("Registering blueprints...")
    # Register blueprints
    from app.auth import auth_bp
    app.register_blueprint(auth_bp)
    print("Auth blueprint registered")

    from app.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp)
    print("Dashboard blueprint registered")

    # Ensure directories exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
    app.config['MODEL_DIR'] = os.path.join(project_root, 'models_storage')
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Model folder: {app.config['MODEL_DIR']}")

    with app.app_context():
        print("Creating database tables...")
        db.create_all()
        print("Database tables created")

    print("=== App Creation Complete ===")
    return app

# Create the app instance
print("Creating app instance...")
app = create_app()
print("App instance created successfully!")