
import torch
from transformers import AutoProcessor, AutoModelForCausalLM 
from ultralytics import YOLO
import pyautogui
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import sys
import os
import cv2
import requests
import numpy as np
from typing import Tuple, List, Union
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from computer_interact.box_annotator import BoxAnnotator 
import easyocr
from paddleocr import PaddleOCR
reader = easyocr.Reader(['en'])
paddle_ocr = PaddleOCR(
    lang='en',  # other lang also available
    use_angle_cls=False,
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    show_log=False,
    max_batch_size=1024,
    use_dilation=True,  # improves accuracy
    det_db_score_mode='slow',  # improves accuracy
    rec_batch_num=1024)

from computer_interact.utils import predict_yolo, int_box_area, remove_overlap_new, get_parsed_content_icon, annotate, get_xywh, get_xyxy

class OmniParser2: 
    
    def __init__(self, logger, config):
        self.config = config
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = self.get_yolo_model(self.config.yolo_model_path).to(self.device)
        self.caption_model_processor = self.get_caption_model_processor(model_path=self.config.caption_model_path)
    def get_caption_model_processor(self, model_path=None):
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        
        return {'model': model, 'processor': processor}
    
    def parse(self):
        pyautogui.screenshot('my_screenshot.png')
        screenshot_path = "my_screenshot.png"
        image = Image.open(screenshot_path)
        self.logger.debug(f"image size:{image.size}")

        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        (text, ocr_bbox), _ = self.check_ocr_box(screenshot_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=True)
        dino_labled_img, label_coordinates, parsed_content_list = self.get_labeled_img(screenshot_path, self.yolo_model, BOX_TRESHOLD = self.config.box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        image.save("my_screenshot_processed.png")
        #self.logger.debug(parsed_content_list)

    def get_yolo_model(self, model_path):
        # Load the model.
        model = YOLO(model_path)
        
        return model

    def get_labeled_img(self, image_source: Union[str, Image.Image], model=None, BOX_TRESHOLD=0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None, scale_img=False, imgsz=None, batch_size=128):
        """Process either an image path or Image object
        
        Args:
            image_source: Either a file path (str) or PIL Image object
            ...
        """
        if isinstance(image_source, str):
            image_source = Image.open(image_source)
        image_source = image_source.convert("RGB") # for CLIP
        w, h = image_source.size
        if not imgsz:
            imgsz = (h, w)
        # print('image size:', w, h)
        xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
        xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
        image_source = np.asarray(image_source)
        phrases = [str(i) for i in range(len(phrases))]

        # annotate the image with labels
        if ocr_bbox:
            ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
            ocr_bbox=ocr_bbox.tolist()
        else:
            self.logger.warning('no ocr bbox!!!')
            ocr_bbox = None

        ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0] 
        xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
        filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
        
        # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
        filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
        # get the index of the first 'content': None
        starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
        filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
        self.logger.debug(f"len(filtered_boxes): {len(filtered_boxes)}, {starting_idx}")

        # get parsed icon local semantics
        time1 = time.time()
        if use_local_semantics:
            caption_model = caption_model_processor['model']
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=prompt,batch_size=batch_size)
            ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
            icon_start = len(ocr_text)
            parsed_content_icon_ls = []
            # fill the filtered_boxes_elem None content with parsed_content_icon in order
            for i, box in enumerate(filtered_boxes_elem):
                if box['content'] is None:
                    box['content'] = parsed_content_icon.pop(0)
            for i, txt in enumerate(parsed_content_icon):
                parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
            parsed_content_merged = ocr_text + parsed_content_icon_ls
        else:
            ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
            parsed_content_merged = ocr_text
        self.logger.debug(f"time to get parsed content:{time.time()-time1}")

        filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

        phrases = [i for i in range(len(filtered_boxes))]
        
        # draw boxes
        if draw_bbox_config:
            annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
        else:
            annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
        
        pil_img = Image.fromarray(annotated_frame)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
        if output_coord_in_ratio:
            label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
            assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

        return encoded_image, label_coordinates, filtered_boxes_elem

    def check_ocr_box(self, image_source: Union[str, Image.Image], display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
        
        if isinstance(image_source, str):
            image_source = Image.open(image_source)
        if image_source.mode == 'RGBA':
            # Convert RGBA to RGB to avoid alpha channel issues
            image_source = image_source.convert('RGB')
        image_np = np.array(image_source)
        w, h = image_source.size
        if use_paddleocr:
            if easyocr_args is None:
                text_threshold = 0.5
            else:
                text_threshold = easyocr_args['text_threshold']
            result = paddle_ocr.ocr(image_np, cls=False)[0]
            coord = [item[0] for item in result if item[1][1] > text_threshold]
            text = [item[1][0] for item in result if item[1][1] > text_threshold]
        else:  # EasyOCR
            if easyocr_args is None:
                easyocr_args = {}
            result = reader.readtext(image_np, **easyocr_args)
            coord = [item[0] for item in result]
            text = [item[1] for item in result]
        if display_img:
            opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            bb = []
            for item in coord:
                x, y, a, b = get_xywh(item)
                bb.append((x, y, a, b))
                cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
            #  matplotlib expects RGB
            #plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
        else:
            if output_bb_format == 'xywh':
                bb = [get_xywh(item) for item in coord]
            elif output_bb_format == 'xyxy':
                bb = [get_xyxy(item) for item in coord]
        return (text, bb), goal_filtering

