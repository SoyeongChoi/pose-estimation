import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms as T
import tensorflow as tf
from pose.models import get_pose_model
from pose.utils.boxes import letterbox, scale_boxes, non_max_suppression, xyxy2xywh
from pose.utils.decode import get_final_preds, get_simdr_final_preds
from pose.utils.utils import setup_cudnn, get_affine_transform, draw_keypoints
from pose.utils.utils import VideoReader, VideoWriter, WebcamStream, FPS

import sys
sys.path.insert(0, 'yolov5')
from yolov5.models.experimental import attempt_load
import onnxruntime as ort

class Pose:
    def __init__(self,
        det_model,
        pose_model,
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.det_model = ort.InferenceSession(det_model) # attempt_load(det_model, map_location=self.device)
        self.det_model = attempt_load(det_model, map_location=self.device)
        self.det_model = self.det_model.to(self.device)

        self.model_name = pose_model
        # self.pose_model = get_pose_model(pose_model)
        self.pose_model_onnx = ort.InferenceSession('simdr.onnx')# get_pose_model(pose_model)
        self.pose_model_tflite = tf.lite.Interpreter('simdr.tflite')
        self.pose_model_tflite.allocate_tensors()
        self.pose_input_details = self.pose_model_tflite.get_input_details()
        self.pose_output_details = self.pose_model_tflite.get_output_details()
        # self.pose_model.load_state_dict(torch.load(pose_model, map_location='cpu'))
        # self.pose_model = self.pose_model.to(self.device)
        # self.pose_model.eval()

        self.patch_size = (192, 256)

        self.pose_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.coco_skeletons = [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]


    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        # img = img.astype(np.float32) / 255.0
        img = img.float() / 255.0
        img = img[None]
        return img

    def box_to_center_scale(self, boxes, pixel_std=200):
        boxes = xyxy2xywh(boxes)
        r = self.patch_size[0] / self.patch_size[1]
        mask = boxes[:, 2] > boxes[:, 3] * r
        boxes[mask, 3] = boxes[mask, 2] / r
        boxes[~mask, 2] = boxes[~mask, 3] * r
        boxes[:, 2:] /= pixel_std
        boxes[:, 2:] *= 1.25
        return boxes

    def predict_poses(self, boxes, img):
        image_patches = []
        for cx, cy, w, h in boxes:
            trans = get_affine_transform(np.array([cx, cy]), np.array([w, h]), self.patch_size)
            img_patch = cv2.warpAffine(img, trans, self.patch_size, flags=cv2.INTER_LINEAR)
            img_patch = self.pose_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)
        # image_patches = image_patches.detach().cpu().numpy()
        # import pdb; pdb.set_trace()
        self.pose_model_tflite.set_tensor(self.pose_input_details[0]['index'], image_patches.detach().cpu().numpy())
        self.pose_model_tflite.invoke()

        pred_x = self.pose_model_tflite.get_tensor(self.pose_output_details[0]['index'])
        pred_y = self.pose_model_tflite.get_tensor(self.pose_output_details[1]['index'])
            
        # pred_x = torch.tensor(self.pose_model_onnx.run([self.pose_model_onnx.get_outputs()[0].name], {self.pose_model_onnx.get_inputs()[0].name: image_patches.detach().cpu().numpy()})[0])
        # pred_y = torch.tensor(self.pose_model_onnx.run([self.pose_model_onnx.get_outputs()[1].name], {self.pose_model_onnx.get_inputs()[0].name: image_patches.detach().cpu().numpy()})[0])
        # preds = [x.detach().numpy() for t in [pred_x, pred_y]]

        # return torch.tensor(preds)# self.pose_model(image_patches)
        # chk2 = self.pose_model(image_patches)
        # return  self.pose_model(image_patches)
        return [torch.tensor(pred_x), torch.tensor(pred_y)] # self.pose_model(image_patches)

    def postprocess(self, pred, img1, img0):
        # import pdb; pdb.set_trace()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0)

        for det in pred:
            if len(det):
                boxes = scale_boxes(det[:, :4], img0.shape[:2], img1.shape[-2:]).cpu()
                boxes = self.box_to_center_scale(boxes)
                outputs = self.predict_poses(boxes, img0)

                if 'simdr' in self.model_name or 'simdir' in self.model_name:
                    coords = get_simdr_final_preds(*outputs, boxes, self.patch_size)
                else:
                    coords = get_final_preds(outputs, boxes)

                draw_keypoints(img0, coords, self.coco_skeletons)

    @torch.no_grad()
    def predict(self, image):

        img = self.preprocess(image)
        pred = self.det_model(img)[0]

        # pred = torch.tensor(self.det_model.run([self.det_model.get_outputs()[0].name], {self.det_model.get_inputs()[0].name: img}))
        self.postprocess(pred, img, image)
        return image


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assests/test.jpg')
    parser.add_argument('--det-model', type=str, default='checkpoints/crowdhuman_yolov5m.pt')
    parser.add_argument('--pose-model', type=str, default='checkpoints/pretrained/simdr_hrnet_w32_256x192.pth')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.4)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    setup_cudnn()
    args = argument_parser()
    pose = Pose(
        args.det_model,
        args.pose_model,
        args.img_size,
        args.conf_thres,
        args.iou_thres
    )

    source = Path(args.source)
    x = torch.randn(1, 3, 256, 192, requires_grad=True)
    train = False
    # torch.onnx.export(pose.pose_model, x, 'simdir.onnx', verbose=False, opset_version=13,
    #                   training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
    #                   do_constant_folding=not train,
    #                   input_names=['images'],
    #                   output_names=['output'],
    #                   dynamic_axes={ 'images': {
    #                         0: 'batch',
    #                         2: 'height',
    #                         3: 'width'},  # shape(1,3,640,640)
    #                     'output': {
    #                         0: 'pred_x',
    #                         1: 'pred_y'}
    #                   }
    #                   )


    if source.is_file() and source.suffix in ['.jpg', '.png']:
        image = cv2.imread(str(source))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = pose.predict(image)
        cv2.imwrite(f"{str(source).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    elif source.is_dir():
        files = source.glob("*.jpg")
        for file in files:
            image = cv2.imread(str(file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = pose.predict(image)
            cv2.imwrite(f"{str(file).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    elif source.is_file() and source.suffix in ['.mp4', '.avi']:
        reader = VideoReader(args.source)
        writer = VideoWriter(f"{args.source.rsplit('.', maxsplit=1)[0]}_out.mp4", reader.fps)
        fps = FPS(len(reader.frames))

        for frame in tqdm(reader):
            fps.start()
            output = pose.predict(frame.numpy())
            fps.stop(False)
            writer.update(output)

        print(f"FPS: {fps.fps}")
        writer.write()

    else:
        webcam = WebcamStream()
        fps = FPS()

        for frame in webcam:
            fps.start()
            output = pose.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps.stop()
            cv2.imshow('frame', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
