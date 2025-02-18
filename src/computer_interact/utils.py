import logging
import yaml
import os.path
#from google.auth.transport.requests import Request
#from google.oauth2.credentials import Credentials
#from google_auth_oauthlib.flow import InstalledAppFlow
from PIL import Image, ImageDraw, ImageColor
from typing import Tuple, List, Union
import cv2
import time
import torch
from torchvision.transforms import ToPILImage
import supervision as sv
from computer_interact.box_annotator import BoxAnnotator 
import numpy as np
from torchvision.ops import box_convert

# Ref - https://medium.com/pythoneers/beyond-print-statements-elevating-debugging-with-python-logging-715b2ae36cd5
def logger_helper(cfg):

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Capture all messages of debug or higher severity

    ### File handler for errors
    # Create a file handler that writes log messages to 'error.log'
    file_handler = logging.FileHandler('error.log') 
    # Set the logging level for this handler to ERROR, which means it will only handle messages of ERROR level or higher
    file_handler.setLevel(logging.ERROR)  

    ### Console handler for info and above
    # Create a console handler that writes log messages to the console
    console_handler = logging.StreamHandler()  
    
    if cfg.debug == True:
        console_handler.setLevel(logging.DEBUG)  
    else:
        # Set the logging level for this handler to INFO, which means it will handle messages of INFO level or higher
        console_handler.setLevel(logging.INFO)  

    ### Set formats for handlers
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
    # Apply the formatter to the file handler 
    file_handler.setFormatter(formatter) 
    # Apply the formatter to the console handler
    console_handler.setFormatter(formatter)  

    ### Add handlers to logger
    # Add the file handler to the logger, so it will write ERROR level messages to 'error.log'
    logger.addHandler(file_handler)  
    # Add the console handler to the logger, so it will write INFO level messages to the console
    logger.addHandler(console_handler)  

    # Now when you log messages, they are directed based on their severity:
    #logger.debug("This will print to console")
    #logger.info("This will also print to console")
    #logger.error("This will print to console and also save to error.log")

    return logger

"""        
def google_api_authenticate(cfg):
    
    if not os.environ.get("GOOGLE_API_CREDS_LOC"):
      raise KeyError("Local file path of credentials.json is missing, please provide it in .env file.") 

    # If modifying these scopes, delete the file token.json.
    SCOPES = cfg['GOOGLE_API']['scopes']

    # Authenticate
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
      creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
      else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0, open_browser=True)
      # Save the credentials for the next run
      with open("token.json", "w") as token:
        token.write(creds.to_json())

    return creds
"""

def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
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

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area

def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
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
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128):
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

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
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

def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h
    
def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp