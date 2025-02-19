
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
        
        return label_coordinates, parsed_content_list 
    
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
        xyxy, logits, phrases = self.predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
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

        ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(ocr_bbox, ocr_text) if self.int_box_area(box, w, h) > 0] 
        xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if self.int_box_area(box, w, h) > 0]
        filtered_boxes = self.remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
        
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
            parsed_content_icon = self.get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=prompt,batch_size=batch_size)
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
            annotated_frame, label_coordinates = self.annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
        else:
            annotated_frame, label_coordinates = self.annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
        
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
                x, y, a, b = self.get_xywh(item)
                bb.append((x, y, a, b))
                cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
            #  matplotlib expects RGB
            #plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
        else:
            if output_bb_format == 'xywh':
                bb = [self.get_xywh(item) for item in coord]
            elif output_bb_format == 'xyxy':
                bb = [self.get_xyxy(item) for item in coord]
        return (text, bb), goal_filtering

    def predict_yolo(self, model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
        """ Use huggingface model to replace the original model
        """
        # model = model['model']
        if scale_img:
            result = model.predict(
            source=image,
            conf=box_threshold,
            imgsz=imgsz,
            iou=iou_threshold, # default 0.7
            )
        else:
            result = model.predict(
            source=image,
            conf=box_threshold,
            iou=iou_threshold, # default 0.7
            )
        boxes = result[0].boxes.xyxy#.tolist() # in pixel space
        conf = result[0].boxes.conf
        phrases = [str(i) for i in range(len(boxes))]

        return boxes, conf, phrases

    def int_box_area(self, box, w, h):
        x1, y1, x2, y2 = box
        int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
        area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
        return area

    def remove_overlap_new(self, boxes, iou_threshold, ocr_bbox=None):
        '''
        ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
        boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

        '''
        assert ocr_bbox is None or isinstance(ocr_bbox, List)

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        def intersection_area(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            return max(0, x2 - x1) * max(0, y2 - y1)

        def IoU(box1, box2):
            intersection = intersection_area(box1, box2)
            union = box_area(box1) + box_area(box2) - intersection + 1e-6
            if box_area(box1) > 0 and box_area(box2) > 0:
                ratio1 = intersection / box_area(box1)
                ratio2 = intersection / box_area(box2)
            else:
                ratio1, ratio2 = 0, 0
            return max(intersection / union, ratio1, ratio2)

        def is_inside(box1, box2):
            # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
            intersection = intersection_area(box1, box2)
            ratio1 = intersection / box_area(box1)
            return ratio1 > 0.80

        # boxes = boxes.tolist()
        filtered_boxes = []
        if ocr_bbox:
            filtered_boxes.extend(ocr_bbox)
        # print('ocr_bbox!!!', ocr_bbox)
        for i, box1_elem in enumerate(boxes):
            box1 = box1_elem['bbox']
            is_valid_box = True
            for j, box2_elem in enumerate(boxes):
                # keep the smaller box
                box2 = box2_elem['bbox']
                if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                    is_valid_box = False
                    break
            if is_valid_box:
                if ocr_bbox:
                    # keep yolo boxes + prioritize ocr label
                    box_added = False
                    ocr_labels = ''
                    for box3_elem in ocr_bbox:
                        if not box_added:
                            box3 = box3_elem['bbox']
                            if is_inside(box3, box1): # ocr inside icon
                                # box_added = True
                                # delete the box3_elem from ocr_bbox
                                try:
                                    # gather all ocr labels
                                    ocr_labels += box3_elem['content'] + ' '
                                    filtered_boxes.remove(box3_elem)
                                except:
                                    continue
                                # break
                            elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                                box_added = True
                                break
                            else:
                                continue
                    if not box_added:
                        if ocr_labels:
                            filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                        else:
                            filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
                else:
                    filtered_boxes.append(box1)
        return filtered_boxes # torch.tensor(filtered_boxes)


    @torch.inference_mode()
    def get_parsed_content_icon(self, filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128):
        # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
        to_pil = ToPILImage()
        if starting_idx:
            non_ocr_boxes = filtered_boxes[starting_idx:]
        else:
            non_ocr_boxes = filtered_boxes
        croped_pil_image = []
        for i, coord in enumerate(non_ocr_boxes):
            try:
                xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
                ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
                cropped_image = image_source[ymin:ymax, xmin:xmax, :]
                cropped_image = cv2.resize(cropped_image, (64, 64))
                croped_pil_image.append(to_pil(cropped_image))
            except:
                continue

        model, processor = caption_model_processor['model'], caption_model_processor['processor']
        if not prompt:
            prompt = "<CAPTION>"

        generated_texts = []
        device = model.device
        for i in range(0, len(croped_pil_image), batch_size):
            batch = croped_pil_image[i:i+batch_size]
            if model.device.type == 'cuda':
                inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
            else:
                inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=3, do_sample=False)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = [gen.strip() for gen in generated_text]
            generated_texts.extend(generated_text)
        
        return generated_texts

    def annotate(self, image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
                text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
        """    
        This function annotates an image with bounding boxes and labels.

        Parameters:
        image_source (np.ndarray): The source image to be annotated.
        boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
        logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
        phrases (List[str]): A list of labels for each bounding box.
        text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

        Returns:
        np.ndarray: The annotated image.
        """
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
        detections = sv.Detections(xyxy=xyxy)

        labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

        box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
        annotated_frame = image_source.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

        label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
        return annotated_frame, label_coordinates

    def get_xywh(self, input):
        x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
        x, y, w, h = int(x), int(y), int(w), int(h)
        return x, y, w, h
        
    def get_xyxy(self, input):
        x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
        x, y, xp, yp = int(x), int(y), int(xp), int(yp)
        return x, y, xp, yp