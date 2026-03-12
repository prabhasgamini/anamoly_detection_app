from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config
import os

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    from app.models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from app.auth import auth_bp
    app.register_blueprint(auth_bp)

    from app.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp)

    # Ensure directories exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
    app.config['MODEL_DIR'] = os.path.join(project_root, 'models_storage')
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

    with app.app_context():
        db.create_all()

    return app

# Create the app instance
app = create_app()