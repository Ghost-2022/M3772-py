from typing import Tuple


class SystemMsg:
    _code = 1000
    SystemError = (_code + 1, 'The system is busy, please try again later')


class AuthMsg:
    _code = 2000
    AuthFailed = (_code + 1, 'The username or password is incorrect')
    LoginRequired = (_code + 2, 'Login required')
    TokenExpired = (_code + 3, 'Token has expired')
    RefreshTokenExpired = (_code + 4, 'Refresh token has expired')
    TokenInvalidation = (_code + 5, 'Token invalidation')
    TokenTypeError = (_code + 6, 'Token type error')
    NoPermissions = (_code + 7, 'The current user does not have permissions')


class UserMsg:
    _code = 3000

    UserNotFound = (_code + 1, 'The user does not exist')
    RoleNotFound = (_code + 2, 'The role does not exist')


class BusinessMsg:
    _code = 4000

    ParametersAreMissing = (_code + 1, f'Necessary parameters are missing')

    @classmethod
    def arg_miss(cls, msg):
        return cls._code + 101, f'Parameters [{msg}] is missing'

    @classmethod
    def arg_err(cls, msg):
        return cls._code + 102, f'Parameters [{msg}] is error'

    @classmethod
    def data_miss(cls, data_name):
        return cls._code + 104, f'Data [{data_name}] is missing'


class DatabaseMsg:
    _code = 5000

    @classmethod
    def error(cls, msg):
        return cls._code + 101, str(msg)


class Error(Exception):
    def __init__(self, code_msg: Tuple[int, str]):
        self.code, self.msg = code_msg

    def __str__(self):
        return self.msg