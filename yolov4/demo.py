import os
import shutil
import time
from pathlib import Path

import cv2
import torch
from numpy import random

from yolov4.utils import LoadImages, non_max_suppression, scale_coords, plot_fire

# from model import *



def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(model, weights, source, out, imgsz, conf_thres, iou_thres, cfg, 
           names, colors=[(255, 30, 0), (50, 0, 255)], device=torch.device('cpu')):
    # Initialize

    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    # Set Dataloader
    vid_path, vid_writer = None, None

    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # print(colors, len(colors), len(names))
    names = load_classes(names)
    if colors is None:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    elif len(colors) < len(names):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time.time()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t2 = time.time()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # if cls == 0:
                        plot_fire(xyxy, im0, clas=cls, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # cv2.imshow(p, im0)
            # if cv2.waitKey(1) == ord('q'):  # q to quit
            #     raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_img:
        pass
        # print('Results saved to %s' % Path(out))

    # print('Done. (%.3fs)' % (time.time() - t0))


# if __name__ == '__main__':
#     weights = '../../yolov4/runs/evolve/weights/best.pt'
#     source  = '../inference/images'
#     out  = '../inference/output'
#     imgsz   = 448
#     conf_thres = 0.35
#     iou_thres = 0.5
#     cfg = 'cfg/yolov4-pacsp.cfg'
#     names = 'data/fire_smoke.names'
#     colors = [(255, 30, 0), (50, 0, 255)]
#     device = torch.device('cpu')


#     # Load model
#     model = Darknet(cfg, imgsz)
#     # try:
#     #     model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
#     # except:
#     #     load_darknet_weights(model, weights[0])

#     model.to(device).eval()

#     # torch.save(model.state_dict(), 'state.pt')
#     # model.load_state_dict(torch.load('state.pt', map_location=device))

#     with torch.no_grad():
#         detect(model, weights, source, out, imgsz, conf_thres, iou_thres, cfg, names, colors, device)
