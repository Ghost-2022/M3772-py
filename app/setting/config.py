from datetime import timedelta
from pathlib import Path
import os

MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PWD = os.environ.get('MYSQL_PWD')
MYSQL_HOST = os.environ.get('MYSQL_HOST')
MYSQL_PORT = os.environ.get('MYSQL_PORT')

class BaseConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent
    FILE_DIR = os.path.join(BASE_DIR, 'static', 'files')
    MODEL_PATH = os.path.join(BASE_DIR, 'static', 'models', 'model_pytorch_200.pth')
    TEMPLATE_DIR = os.path.join(BASE_DIR, 'static')
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SECRET_KEY = 'nfr2xxch$5nd6@ue2i%arcqxyl@gv@0t'
    SESSION_TYPE = 'sqlalchemy'
    SESSION_SQLALCHEMY_TABLE = 'sessions'
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    # SQLALCHEMY_DATABASE_URI = f'mysql://{MYSQL_USER}:{MYSQL_PWD}@{MYSQL_HOST}:{MYSQL_PORT}/m3772'
    SQLALCHEMY_DATABASE_URI = f'mysql://jc:q2eh$i7epb#lj#h&@101.42.16.61:3356/m3772'
    SALT = '$2b$12$c1AOB11G.dX732PGBt4jcu'


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    DEBUG = True