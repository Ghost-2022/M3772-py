
import time

from flask import request, current_app, session, g

from app.api.view_base import ViewBase
from app.common.db import db
from app.common.decorator import check_login_perm
from app.common.error import Error, AuthMsg, BusinessMsg, DatabaseMsg
from app.common.utils import parse_jwt_token, gen_token
from app.models.account import User


class LoginView(ViewBase):
    def post(self):
        username = request.json.get('username')
        password = request.json.get('password')
        user = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
        if not user:
            current_app.logger.warning(f'This user [{username}] does not exist')
            raise Error(AuthMsg.AuthFailed)
        if not user.verify_pwd(password):
            current_app.logger.warning(f'The user [{username}] has an incorrect password')
            raise Error(AuthMsg.AuthFailed)
        expired_time = int(time.time()) + 3600 * 2400
        token, refresh_token = gen_token(user.id, current_app.config['SECRET_KEY'], expired_time)
        session[f'user-{user.id}'] = {'expiredTime': expired_time, 'token': token}
        return {'token': token, 'refreshToken': refresh_token, 'userId': user.id}


class LogoutView(ViewBase):
    @check_login_perm()
    def get(self):
        session.pop(f'user-{g.user.id}', None)


class RefreshTokenView(ViewBase):
    def post(self):
        refresh_token = request.json.get('refreshToken')
        token_data = parse_jwt_token(refresh_token, current_app.config['SECRET_KEY'])
        if not session.get(f'user-{token_data["userId"]}'):
            raise Error(AuthMsg.TokenInvalidation)

        if token_data['expiredTime'] < time.time():
            raise Error(AuthMsg.RefreshTokenExpired)

        if token_data.get('purpose') != 'refresh_token':
            raise Error(AuthMsg.TokenTypeError)

        expired_time = int(time.time()) + 3600 * 24
        token, refresh_token = gen_token(token_data['userId'], current_app.config['SECRET_KEY'], expired_time)
        session[f'user-{token_data["userId"]}'] = {'expiredTime': expired_time, 'token': token}
        return {'token': token, 'refreshToken': refresh_token}


class UserInfoView(ViewBase):
    @check_login_perm()
    def get(self):
        data = g.user.to_dict()
        return data


class RegisterView(ViewBase):
    def post(self):
        username = request.json.get('username')
        password = request.json.get('password')
        if not username or not password:
            raise Error(BusinessMsg.arg_err('username or password miss'))
        user: User = User(username=username, gender=1, is_admin=0,
                          avatar='https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif')
        user.password = user.encrypt_pwd(password)
        try:
            db.session.add(user)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise Error(DatabaseMsg.error('系统繁忙'))