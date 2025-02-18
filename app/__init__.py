import os

import pymysql
from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate


from app.common.db import db
# from app.common.session import session
from app.setting import env_map
from app.common.bcrypt import bcrypt

pymysql.install_as_MySQLdb()


def create_app():
    app = Flask(__name__, template_folder='static/')
    env = os.getenv('APP_ENV', 'development')
    app.config.from_object(env_map.get(env))
    app.json.ensure_ascii = False
    app.config['SESSION_SQLALCHEMY'] = db

    db.init_app(app)
    # session.init_app(app)
    bcrypt.init_app(app)

    CORS(app, origins='*',  resources=r'/*')
    Migrate(app, db)
    from .api import api_bp
    app.register_blueprint(api_bp)
    return app