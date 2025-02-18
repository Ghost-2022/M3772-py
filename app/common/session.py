from flask_session.sqlalchemy import SqlAlchemySessionInterface
from flask_session import Session

class FixedSqlAlchemySessionInterface(SqlAlchemySessionInterface):
    def __init__(self, app, db, table, key_prefix, use_signer=False,
                 permanent=True):
        """
        Assumption: the way I use it, db is always a valid instance
        at this point.
        """
        if table not in db.metadata:
            super().__init__(app, db, table, None, None, None, key_prefix, use_signer, permanent, 32)
            db.session_ext_session_model = self.sql_session_model
        else:
            # print( "`sessions` table already exists..." )

            self.db = db
            self.key_prefix = key_prefix
            self.use_signer = use_signer
            self.permanent = permanent
            self.has_same_site_capability = hasattr(self, "get_cookie_samesite")
            self.sql_session_model = db.session_ext_session_model


class FixedSession(Session):
    def _get_interface(self, app):
        config = app.config.copy()

        if config['SESSION_TYPE'] != 'sqlalchemy':
            return super()._get_interface(app)

        else:
            config.setdefault('SESSION_PERMANENT', True)
            config.setdefault('SESSION_USE_SIGNER', False)
            config.setdefault('SESSION_KEY_PREFIX', 'session:')
            config.setdefault('SESSION_SQLALCHEMY', None)
            config.setdefault('SESSION_SQLALCHEMY_TABLE', 'sessions')

            return FixedSqlAlchemySessionInterface(
                app, config['SESSION_SQLALCHEMY'],
                config['SESSION_SQLALCHEMY_TABLE'],
                config['SESSION_KEY_PREFIX'], config['SESSION_USE_SIGNER'],
                config['SESSION_PERMANENT'])


# session = FixedSession()
session = Session()