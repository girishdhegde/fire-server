import numpy as np
import os
import flask
from flask import Flask,request,jsonify,url_for,render_template
import torch
import cv2
from yolov4.demo import detect
from yolov4.model import *


weights = 'yolov4/weights/yolov4-pacsp.pt'
# weights = '../yolov4/weights/pacsp/fire3.pt'
source  = 'static'
out  = 'static/outputs'
imgsz   = 448
conf_thres = 0.35
iou_thres = 0.5
cfg = 'yolov4/cfg/yolov4-pacsp.cfg'
names = 'yolov4/data/fire_smoke.names'
colors = [(255, 30, 0), (50, 0, 255)]
device = torch.device('cpu')

torch.hub.download_url_to_file('https://www.dropbox.com/s/jlep1pe8xt9quxf/fire3.pt?dl=1', weights)

# Load model
model = Darknet(cfg, imgsz)
try:
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
except:
    load_darknet_weights(model, weights)

model.to(device).eval()

# torch.save(model.state_dict(), 'state.pt')
# model.load_state_dict(torch.load('state.pt', map_location=device))


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    if flask.request.method == "GET":
        return render_template("index.html")
    else:
        # shutil.rmtree('./static')
        # os.mkdir('./static')
        f = request.files["image"]
        fmat =f.filename.split('.')[-1]
        path1 = f'./static/img.{fmat}'
        path2 = f'./static/outputs/img.{fmat}'

        f.save(path1)

        with torch.no_grad():
            detect(model, weights, source, out, imgsz, conf_thres, iou_thres, cfg, names, colors, device)
        # cv2.imshow('ing', path2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return render_template("upload.html", img1=path1, img2=path2)
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)
