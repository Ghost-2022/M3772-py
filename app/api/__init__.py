from flask import Blueprint, jsonify, current_app

from app.common.error import Error
from .account_api import LoginView, LogoutView, RefreshTokenView, UserInfoView, RegisterView


def error_handler(err: Error):
    current_app.logger.info(f'Error: {err}')
    return jsonify({'code': err.code, 'msg': err.msg})


api_bp = Blueprint('api', __name__, url_prefix='/api/', )

api_bp.add_url_rule("/login/", endpoint=None, view_func=LoginView.as_view("login"))
api_bp.add_url_rule("/register/", endpoint=None, view_func=RegisterView.as_view("register"))
api_bp.add_url_rule("/logout/", endpoint=None, view_func=LogoutView.as_view("logout"))
api_bp.add_url_rule("/refresh-token/", endpoint=None, view_func=RefreshTokenView.as_view("refresh-token"))
api_bp.add_url_rule("/user-info/", endpoint=None, view_func=UserInfoView.as_view("get-userinfo"))
# api_bp.add_url_rule("/get-user-address/", endpoint=None, view_func=UserAddresses.as_view("user-addresses"))
