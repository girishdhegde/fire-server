from flask import Flask, request, render_template
import torch
from yolov4.demo import detect
from yolov4.model import Darknet
import base64

# weights = 'yolov4/weights/yolov4-pacsp.pt'
weights = './pretrained.pt'
# source  = 'static'
# out  = 'static/outputs'
imgsz   = 448
conf_thres = 0.35
iou_thres = 0.5
cfg = 'yolov4/cfg/yolov4-pacsp.cfg'
names = ['fire', 'smoke']
colors = [(255, 30, 0), (50, 0, 255)]
device = torch.device('cpu')

torch.hub.download_url_to_file('https://www.dropbox.com/s/2f3jcqy2qxae4aj/pretrained.pt?dl=1', weights)

# Load model
model = Darknet(cfg, imgsz)

model.load_state_dict(torch.load(weights, map_location=device))

model.to(device).eval()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        # shutil.rmtree('./static')
        # os.mkdir('./static')
        f = request.files["image"]
        fmat =f.filename.split('.')[-1]
        path1 = f'./static/img.{fmat}'
        path2 = f'./static/outputs/img.{fmat}'

        # path1 = f'./static/{f.filename}'
        # path2 = f'./static/outputs/{f.filename}'
        f.save(path1)

        detect(model, path1, path2, imgsz, conf_thres, iou_thres, names, colors, device)

        return render_template("upload.html", img1=path1, img2=path2)

@app.route('/mobile', methods=['GET', 'POST'])
def mobile():
    if request.method == "GET":
        return render_template("index.html")
    else:
        # shutil.rmtree('./static')
        # os.mkdir('./static')
        f = request.files["image"]
        fmat =f.filename.split('.')[-1]
        path1 = f'./static/img.{fmat}'
        path2 = f'./static/outputs/img.{fmat}'

        # path1 = f'./static/{f.filename}'
        # path2 = f'./static/outputs/{f.filename}'
        f.save(path1)
        img = cv2.imread(path2) # reads the PIL image
        retval, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer)
        return img_base64

if __name__ == "__main__":
    app.run(debug=False)
