from app.common.db import db
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import text
from app.common.utils import datetime2str, convert_to_camel_case


class ModelMixin:
    is_del = mapped_column(db.Boolean(), comment='是否删除', server_default=text('0'))
    created_time: Mapped[datetime] = mapped_column(db.DateTime, default=datetime.now,
                                                   server_default=text('CURRENT_TIMESTAMP'),
                                                   comment='创建时间')
    updated_time: Mapped[datetime] = mapped_column(db.DateTime,
                                                   server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'),
                                                   comment='更新时间')

    @classmethod
    def get_fields(cls, is_all=False, show_id=False):
        cols = cls.__table__.columns.keys()
        if not is_all:
            cols.remove('created_time')
            cols.remove('updated_time')
            cols.remove('is_del')
        if not show_id:
            cols.remove('id')
        return cols

    def to_dict(self):
        cols = self.get_fields(show_id=True)
        data = {}
        for k in cols:
            if k in ('created_time', 'updated_time'):
                continue

            val = getattr(self, k)
            if isinstance(val, datetime):
                val = datetime2str(val)
            elif k in ('id',):
                val = str(val)
            data[convert_to_camel_case(k)] = val
        return data
