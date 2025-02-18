from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.db import db
from app.models.model_base import ModelMixin

class IdentificationRecords(db.Model, ModelMixin):
    __tablename__ = 'identification_records'
    id: Mapped[int] = mapped_column(db.Integer, primary_key=True)
    user_id = mapped_column(db.Integer, ForeignKey('users.id'))
    file_path = mapped_column(db.String(255))
    result = mapped_column(db.Integer)
    user = relationship("User", back_populates="identification_records")

