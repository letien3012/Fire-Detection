#Thêm thư viện
from flask import Flask, render_template, Response,jsonify,request,session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
from ultralytics import YOLO
import math
import os
import cv2
from datetime import datetime

#Khởi tạo Flask Server Backend
app = Flask(__name__)

#Áp dụng Flask CORS
app.config['SECRET_KEY'] = 'leminhtien'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SAVE_FORDER'] = 'static/detect'
#Hàm nhận dạng video
def video_detection(path_x):
    video_capture = path_x
    #Tạo đối tượng video
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
    if (path_x == 0):
        model=YOLO("model/Fire.pt")
        classNames = ["Fire"]
    else: 
        model=YOLO("model/Fire_Smoke.pt")
        classNames = ["Smoke","Fire"]
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                p1, p2 = (x1, y1), (x2, y2)
                cv2.rectangle(img, p1, p2, (255,0,255), thickness=2, lineType=cv2.LINE_AA)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                if label:
                    # tf = max(lw - 1, 1)  # font thickness
                    w, h = cv2.getTextSize(label , 0, fontScale=1, thickness=2)[0]  # text width, height
                    outside = p1[1] - h - 3 >= 0  # label fits outside box
                    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                    cv2.rectangle(img, p1, p2, [255,0,255], -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1, [255,255,255],
                                thickness=1, lineType=cv2.LINE_AA)
        if (path_x == 0): cv2.putText(img, datetime.now().strftime("%H:%M:%S"), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              1,  (255, 255, 255), 1, cv2.LINE_AA)    
        yield img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()

#Nhận tệp video đầu vào từ người dùng bằng FlaskForm
class UploadFileForm(FlaskForm):
    #Tệp video được tải lên được lưu trong biến của trường FileField
    file = FileField("File",validators=[InputRequired()])
    #Kiểm tra dịnh dạng video
    submit = SubmitField("Run")


def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/', methods=['GET','POST'])
def home():
    # Tạo biểu mẫu để upload file
    form = UploadFileForm()
    if form.validate_on_submit():
        # Lưu đường dẫn của file được upload và lưu file vào đường dẫn
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  
        # Sử dụng bộ nhớ của session để lưu đường dẫn
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('indexproject.html', form=form)

@app.route('/video')
def video():
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

# Hiển thị hình ảnh của webcam
@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

#Start Backend
if __name__ == "__main__":
    app.run(debug=True)

cv2.destroyAllWindows()