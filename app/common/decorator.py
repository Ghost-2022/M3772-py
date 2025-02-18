import functools
import time

from flask import request, current_app, g, session

from app.common.db import db
from app.common.utils import parse_jwt_token
from app.common.error import Error, AuthMsg
from app.models.account import User


def check_login_perm(perm_key=None):
    def wrapper(f):

        @functools.wraps(f)
        def inner(*args, **kwargs):
            token = request.headers.get('Authorization')
            if token:
                token = token.split(' ')[-1]
                data = parse_jwt_token(token, current_app.config['SECRET_KEY'])
                sess_token = session.get(f'user-{data["userId"]}')
                # if not sess_token:
                #     raise Error(AuthMsg.TokenInvalidation)
                if data.get('purpose') != 'auth':
                    raise Error(AuthMsg.TokenTypeError)
                if data['expiredTime'] <= time.time():
                    raise Error(AuthMsg.TokenExpired)
                user: User = db.session.execute(db.select(User).filter_by(id=data['userId'])).scalar_one_or_none()
                g.user = user
                if perm_key and perm_key not in user.get_perm_key():
                    raise Error(AuthMsg.NoPermissions)
            else:
                raise Error(AuthMsg.LoginRequired)
            return f(*args, **kwargs)
        return inner
    return wrapper