from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import io
from io import StringIO
from PIL import Image
import base64
import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
from datetime import datetime
from mtcnn import MTCNN


detector = MTCNN()
color = (100, 250, 25)
font = cv2.FONT_HERSHEY_SIMPLEX
scale_percent = 80

faces = {}
#
# for f in os.listdir('faces'):
#     if 'png' in f:
#         name = f.split('.')[0][0:-2]
#         if name not in faces:
#             faces[name] = [cv2.cvtColor(cv2.imread(str('faces/' + f), 0), cv2.COLOR_BGR2RGB)]
#         else:
#             faces[name].append(cv2.cvtColor(cv2.imread(str('faces/' + f), 0), cv2.COLOR_BGR2RGB))
# print(faces)


for f in os.listdir('faces'):
    if 'png' in f:
        faces[f.split('.')[0]] = cv2.cvtColor(cv2.imread(str('faces/' + f), 0), cv2.COLOR_BGR2RGB)

app = Flask(__name__)

socketio = SocketIO(app)
app.config['SECRET_KEY'] = 'secret!'


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@socketio.on('rec_faces_output')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)
    pimg = find_faces_old(np.array(pimg))
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    imgencode = cv2.imencode('.jpg', frame)[1]
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData
    emit('response_back', stringData)


if __name__ == '__main__':
    socketio.run(app)


def find_faces_new(img):
    global faces
    people = faces.keys()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    loc = detector.detect_faces(img)
    face_result = {}

    if len(loc) != 0:
        print('__found__')
        for face in loc:
            start_point = face['box'][0], face['box'][1]
            text_start_point = face['box'][0], face['box'][1] - 15
            end_point = face['box'][0] + face['box'][2], face['box'][1] + face['box'][3]

            face_crop = img[face['box'][1]:face['box'][1] + face['box'][3],
                            face['box'][0]:face['box'][0] + face['box'][2]]
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            for person in faces.keys():
                for person_image in faces[person]:
                    compare = DeepFace.verify(face_crop, person_image, model_name='Facenet512', enforce_detection=False,
                                              distance_metric='euclidean_l2')
                    if compare['verified']:
                        face_result[person] = compare['distance']
                        break

            min_dis = 100
            best_name = ''
            if face_result != {}:
                print(face_result)
                for key, value in face_result.items():
                    if value < min_dis:
                        min_dis = value
                        best_name = key
                img = cv2.rectangle(img, start_point, end_point, color, 4)
                img = cv2.putText(img, best_name, text_start_point, font, 1, color, 2)
    return img


def find_faces_old(image):
    people = {}
    for f in os.listdir('faces'):
        if 'png' in f:
            people[f.split('.')[0][0:-2]] = 'absent'
    results = pd.DataFrame(columns=['Presence', 'Identification D.', 'Photo'])
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    loc = detector.detect_faces(img)
    if len(loc) != 0:
        for face in loc:
            start_point = face['box'][0], face['box'][1]
            text_start_point = face['box'][0], face['box'][1] - 15
            end_point = face['box'][0] + face['box'][2], face['box'][1] + face['box'][3]

            face_crop = img[face['box'][1]:face['box'][1] + face['box'][3],
                        face['box'][0]:face['box'][0] + face['box'][2]]
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            verified = 0
            for key, value in faces.items():
                result = DeepFace.verify(face_crop, value, model_name='Facenet512', enforce_detection=False,
                                         distance_metric='euclidean_l2')
                if result['verified']:
                    img = cv2.rectangle(img, start_point, end_point, color, 4)
                    img = cv2.putText(img, key + ', d=' + str(round(result['distance'], 3)), text_start_point, font, 1,
                                      color, 2)
                    verified += 1
                    temp_name = key[0:-2]
                    temp_dist = round(result['distance'], 3)
                    temp = pd.DataFrame({'Presence': 'True', 'Identification D.': temp_dist}, index=[temp_name])
                    results = pd.concat([results, temp])
                    break

            if verified == 0:
                img = cv2.rectangle(img, start_point, end_point, color, 2)
                img = cv2.putText(img, 'Not defined', text_start_point, font, 1, color, 2)
                temp = pd.DataFrame({'Presence': ['True'], 'Identification D.': ['Not defined']})
                results = pd.concat([results, temp])

    for key, value in people.items():
        if key not in results.index.tolist():
            temp = pd.DataFrame({'Presence': 'False'}, index=[key])
            results = pd.concat([results, temp])
    results.to_csv('static/reports/' + str(datetime.now()).replace(':', '.') + '.csv')
    return img