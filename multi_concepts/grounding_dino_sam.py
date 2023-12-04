import torch
import os
import sys
import argparse

sys.path.insert(0, os.path.join(sys.path[0], 'thirdparties/GroundingDINO'))

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import numpy as np
import cv2
from tqdm.auto import tqdm

from openai import OpenAI
import base64
import requests


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def gpt4v_captioning(img_dir):

    headers = {
        "Content-Type": "application/json", "Authorization":
        f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    images = [encode_image(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)]
    prompt = "All these images are from one individual. What garments are in these images? \
    Please list all the garments in the form of 'garment1|description, garment2|description, ...'\
        separated with comma, thus the reponse could be directly used by python array loader."

    payload = {
        "model": "gpt-4-vision-preview", "messages":
        [{"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]}], "max_tokens": 300
    }
    for image in images:
        payload["messages"][0]["content"].append({
            "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    result = response.json()['choices'][0]['message']['content']
    print(result)

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help="input image folder")
    parser.add_argument('--out_dir', type=str, required=True, help="output mask folder")
    opt = parser.parse_args()

    os.makedirs(f"{opt.out_dir}/mask", exist_ok=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # paths
    GroundingDINO_dir = "thirdparties/GroundingDINO"
    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GroundingDINO_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        GroundingDINO_dir, "weights/groundingdino_swint_ogc.pth"
    )
    SAM_CHECKPOINT_PATH = os.path.join(GroundingDINO_dir, "weights/sam_vit_h_4b8939.pth")
    SAM_ENCODER_VERSION = "vit_h"

    # load models
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
    )
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    BOX_TRESHOLD = 0.50
    TEXT_TRESHOLD = 0.50

    CLASSES = [item.split("|")[0].strip().replace("t-","") for item in gpt4v_captioning(opt.in_dir).split(",")]

    print(CLASSES)

    for img_name in tqdm(os.listdir(opt.in_dir)):

        img_path = os.path.join(opt.in_dir, img_name)

        image = cv2.imread(img_path)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        mask_dict = {}

        for mask, cls_id in zip(detections.mask, detections.class_id):
            if cls_id is not None:
                mask_dict[cls_id] = mask_dict.get(cls_id, []) + [mask]

        for cls_id, masks in mask_dict.items():
            mask = np.stack(masks).sum(axis=0)
            mask = (mask > 0).astype(np.uint8) * 255
            cv2.imwrite(f"{opt.out_dir}/mask/{img_name[:-4]}_{CLASSES[cls_id]}.png", mask)
