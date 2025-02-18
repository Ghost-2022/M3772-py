from flask.cli import with_appcontext

from app.common.db import db
from app.models.account import User


# noinspection PyArgumentList
@with_appcontext
def init_db():
    # noinspection PyArgumentList
    user: User = User(username='admin', gender=1, is_admin=1,
                      avatar='https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif')
    user.password = user.encrypt_pwd('123456')
    db.session.add(user)

    db.session.commit()
