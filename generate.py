# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import time
import setting
import os


def generate_captcha_text():
    captcha_text = []
    for i in range(setting.MAX_CAPTCHA):
        c = random.choice(setting.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)


def generate_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = generate_captcha_text()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


if __name__ == '__main__':
    count = 100
    path = setting.TRAIN_DATASET_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        now = str(int(time.time()))
        text, image = generate_captcha_text_and_image()
        filename = text + '_' + now + '.png'
        image.save(path + os.path.sep + filename)
        print('saved %d : %s' % (i + 1, filename))
