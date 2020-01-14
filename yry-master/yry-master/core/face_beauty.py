import requests
import cv2
import json
import base64


def beauty_image(image):
    url = 'https://api-cn.faceplusplus.com/facepp/v1/beautify'
    params = {
        'api_key': 'AbnawESWT1tNA6mI9PfQWNAF4iTeiza-',
        'api_secret': 'n4GQyk1XMjcdN_F3Nq3dwbXR6qonbodH',
    }
    file = {'image_file': open(image, 'rb')}
    r = requests.post(url=url, files=file, data=params)

    if r.status_code == requests.codes.ok:
        return r.content.decode('utf-8')
    else:
        return r.content
