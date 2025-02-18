from sqlalchemy.orm import Mapped, mapped_column

from app.common.bcrypt import bcrypt
from app.common.db import db
from app.models.model_base import ModelMixin


class User(db.Model, ModelMixin):
    # __table_args__ = {'extend_existing': True}
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    username: Mapped[str] = mapped_column(db.String(50), unique=True)
    password: Mapped[str] = mapped_column(db.String(255))
    gender: Mapped[int] = mapped_column(db.Boolean(), nullable=False, comment='性别 0：女；1：男')
    active: Mapped[int] = mapped_column(db.Boolean(), default=True)
    avatar: Mapped[str] = mapped_column(db.String(255), nullable=False, comment='头像')
    is_admin: Mapped[int] = mapped_column(db.Boolean(), default=False)

    @staticmethod
    def encrypt_pwd(pwd: str) -> str:
        return bcrypt.generate_password_hash(pwd)

    def verify_pwd(self, pwd: str) -> bool:
        return bcrypt.check_password_hash(self.password, pwd)

    def to_dict(self):
        data = {
            'id': self.id,
            'username': self.username,
            'gender': 1 if self.gender else 0,
            'active': 1 if self.active else 0,
            'avatar': self.avatar
        }
        return data

    def check_collect(self, club_id: int) -> bool:
        shop_ids = [item.shop_id for item in self.user_collect]
        return club_id in shop_ids

    def check_like(self, club_id: int) -> bool:
        shop_ids = [item.shop_id for item in self.user_likes]
        return club_id in shop_ids
