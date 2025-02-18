from PIL import Image
from flask import request, g
from sqlalchemy import func

from app.api.view_base import ViewBase
from app.common.cnn_model import device, model
from app.common.db import db
from app.common.decorator import check_login_perm
from app.common.error import Error, BusinessMsg, DatabaseMsg
from app.common.utils import predict_image
from app.models.business import IdentificationRecords


class IdentificationView(ViewBase):
    @check_login_perm()
    def post(self):
        file = request.files.get('file')
        if file is None:
            raise Error(BusinessMsg.arg_err('未上传图片'))
        img = Image.open(file.stream)
        result = predict_image(img, device, model)
        record = IdentificationRecords(user_id=g.user.id, file_path='', result=result)
        try:
            db.session.add(record)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise Error(DatabaseMsg.error('系统繁忙'))

# summary
class IdentificationSummaryView(ViewBase):
    @check_login_perm()
    def get(self):
        records = db.session.execute(
            db.select(IdentificationRecords.result, func.count(IdentificationRecords.id).label('cnt')
                      ).filter_by(user_id=g.user.id).group_by(IdentificationRecords.result)).scalar_one_or_none()
        print(records)
