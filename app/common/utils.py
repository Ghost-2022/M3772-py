import datetime
from typing import Tuple

import jwt
import torch
from PIL import Image
from loguru import logger
from torchvision import transforms

from app.common.error import Error, AuthMsg, SystemMsg


def gen_jwt_token(data: dict, secret: str) -> str:
    try:
        encoded_jwt = jwt.encode(data, secret, algorithm="HS256")
    except Exception as e:
        msg = f'Data encryption failed: {e}'
        logger.error(msg)
        raise Error(SystemMsg.SystemError)
    return encoded_jwt


def parse_jwt_token(token, secret: str) -> dict:
    try:
        data = jwt.decode(token, secret, algorithms=["HS256"])
    except Exception as e:
        msg = f'Token parsing failed: {e}'
        logger.warning(msg)
        raise Error(AuthMsg.TokenInvalidation)
    return data


def gen_token(user_id: int, secret_key: str, expired_time: int) -> Tuple[str, str]:
    token_data = {'userId': user_id, 'expiredTime': expired_time, 'purpose': 'auth'}
    token = gen_jwt_token(token_data, secret_key)
    refresh_token_data = {'userId': user_id, 'expiredTime': expired_time + 3600 * 12, 'purpose': 'refresh_token'}
    refresh_token = gen_jwt_token(refresh_token_data, secret_key)
    return token, refresh_token


def date2str(date: datetime.date) -> str:
    return date.strftime("%Y-%m-%d") if date else ''


def datetime2str(date: datetime.datetime) -> str:
    return date.strftime("%Y-%m-%d %H:%M:%S") if date else ''


def convert_to_camel_case(text):
    words = text.split('_')
    camel_case_words = [words[0].lower()] + [word.capitalize() for word in words[1:]]
    camel_case_text = ''.join(camel_case_words)
    return camel_case_text


def username_encry(username: str) -> str:
    return f"{username[0]}*****{username[-1]}"


def predict_image(img: Image, device, model):
    # 加载图片
    image = img.convert('RGB')  # 确保图像是RGB格式
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 根据训练时的正则化方式设置
    ])
    image = transform(image).unsqueeze(0).to(device)  # 添加批次维度并转换为tensor

    # 进行预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # 选择概率最大的类别
        predicted_class = predicted.item()

    return predicted_class
