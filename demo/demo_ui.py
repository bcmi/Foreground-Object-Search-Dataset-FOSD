import sys,os
PROJ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_dir)
import cv2
import math
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from network.networks import StudentModel
from config.config import Config
import datetime
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms

IMAGE_NET_MEAN = [0.5, 0.5, 0.5]
IMAGE_NET_STD  = [0.5, 0.5, 0.5]

def fill_box_with_specified_pixel(bg_im, query_box, fill_value):
    x1, y1 = query_box[0]
    x2, y2 = query_box[1]
    bg_im = np.array(bg_im)
    bg_im[y1:y2, x1:x2] = fill_value
    bg_im = Image.fromarray(bg_im)
    return bg_im

def padding_to_square(src_img, pad_pixel=255):
    src_h, src_w = src_img.shape[:2]
    if src_h == src_w:
        return src_img
    if src_w > src_h:
        pad_w = 0
        pad_h = src_w - src_h
    else:
        pad_w = src_h - src_w
        pad_h = 0

    pad_y1 = int(pad_h // 2)
    pad_y2 = int(pad_h - pad_y1)
    pad_x1 = int(pad_w // 2)
    pad_x2 = int(pad_w - pad_x1)

    if len(src_img.shape) == 3:
        pad_im = np.pad(src_img, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0,0)),
                        'constant', constant_values=pad_pixel)
    else:
        pad_im = np.pad(src_img, ((pad_y1, pad_y2), (pad_x1, pad_x2)),
                        'constant', constant_values=pad_pixel)
    return pad_im

def prepare_input(bg_im, fg_im, bbox_points, cfg):

    transform_withresize = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
    bg_im = fill_box_with_specified_pixel(bg_im, bbox_points, cfg.fill_pixel)
    bg_t = transform_withresize(bg_im)

    bg_ori_w, bg_ori_h = bg_im.size
    assert bg_ori_h == 512 and bg_ori_w == 512
    ori_x1, ori_y1 = bbox_points[0]
    ori_x2, ori_y2 = bbox_points[1]
    query_box = torch.tensor([ori_x1 / bg_ori_w, ori_y1 / bg_ori_h, ori_x2 / bg_ori_w, ori_y2 / bg_ori_h]) * cfg.image_size
    query_box = torch.round(query_box)
    query_box = query_box.float()

    ori_tar_w = ori_x2 - ori_x1
    ori_tar_h = ori_y2 - ori_y1

    add_w = int(ori_tar_w * (math.sqrt(2) - 1) / 2)
    add_h = int(ori_tar_h * (math.sqrt(2) - 1) / 2)
    new_y1 = max(0, ori_y1 - add_h)
    new_y2 = min(bg_ori_h, ori_y2 + add_h)
    new_x1 = max(0, ori_x1 - add_w)
    new_x2 = min(bg_ori_w, ori_x2 + add_w)
    new_box = torch.tensor([new_x1 / bg_ori_w, new_y1 / bg_ori_h, new_x2 / bg_ori_w, new_y2 / bg_ori_h]) * cfg.image_size
    new_box = torch.round(new_box)
    new_box = new_box.float()

    pad_fg = padding_to_square(fg_im.copy(), cfg.pad_pixel)
    pad_fg = Image.fromarray(pad_fg)
    fg_t = transform_withresize(pad_fg)

    sample = {
        'bg': bg_t.unsqueeze(0),
        'fg': fg_t.unsqueeze(0),
        'query_box': query_box.unsqueeze(0),
        'crop_box': new_box.unsqueeze(0),
    }
    return sample


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

def inference(source_image,
              masked_image,
              fg_image,
              points,
              save_dir="./demo/results"
    ):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    config_file = os.path.join(PROJ_dir, 'config/config_rfosd.yaml')
    cfg = Config(config_file)
    model = StudentModel(cfg).to(device).eval()
    ckpt_dir = os.path.join(PROJ_dir, 'checkpoints')
    assert os.path.exists(ckpt_dir), ckpt_dir
    weight_epoch = 'rfosd.pth'
    weight_file = os.path.join(ckpt_dir, weight_epoch)
    assert os.path.exists(weight_file), weight_file
    print('load weights ', weight_file)
    weights = torch.load(weight_file)
    model.load_state_dict(weights, strict=True)

    sample = prepare_input(source_image, fg_image, points, cfg)

    image_with_bbox = preprocess_image(masked_image, device)
    fg_image = preprocess_image(fg_image, device)

    bg_im = sample['bg'].to(device)
    fg_im = sample['fg'].to(device)
    q_box  = sample['query_box'].to(device)
    c_box  = sample['crop_box'].to(device)
    print('Bbox left-top point:', q_box[0, :2])
    print('Bbox right-bottom point:', q_box[0, 2:])
    print('Cropped left-top point:', c_box[0, :2])
    print('Cropped right-bottom point:', c_box[0, 2:])
    output = model(bg_im, fg_im, q_box, c_box)[-1].item()

    save_result = torch.cat([
        image_with_bbox * 0.5 + 0.5,
        torch.ones((1,3,512,25)).cuda(),
        fg_image * 0.5 + 0.5,
        torch.ones((1,3,512,25)).cuda(),
    ], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S") + '_%.3f' % output
    save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))
    return output

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""
        # Compatibility Predictor: [DiscoFOS](https://arxiv.org/abs/2308.04990)
        """)

    with gr.Tab(label="Image"):
        with gr.Row():
            original_image = gr.State(value=None)
            mask = gr.State(value=None)
            selected_points = gr.State([])
            length = 720
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Upload Background</p>""")
                canvas = gr.Image(type="numpy", label="Background", show_label=True, height=length, width=length)
                
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Click Points</p>""")
                input_image = gr.Image(type="numpy", label="Click Points", show_label=True, height=length, width=length)
                undo_button = gr.Button("Undo point", scale=1, min_width=length)
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Upload Foreground</p>""")
                fg_input_image = gr.Image(type="numpy", label="Foreground", show_label=True, height=length, width=length)
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Composite Image</p>""")
                comp_image = gr.Image(type="numpy", label="Composite Image", show_label=True, height=length, width=length)
                run_button = gr.Button("Run", scale=1, min_width=length)
        with gr.Row():
            score = gr.Number(label="Predicted Score", precision=3, scale=1, min_width=length*2)

    def store_img(img):
        image = img
        image = Image.fromarray(image)
        image = image.resize((512,512), PIL.Image.BILINEAR)
        image = np.array(image)
        return image, [], image
    canvas.upload(
        store_img,
        [canvas],
        [original_image, selected_points, input_image]
    )

    def get_point(img, sel_pix, evt: gr.SelectData):
        sel_pix.append(evt.index)
        points = []
        for idx, point in enumerate(sel_pix):
            points.append(tuple(point))
            if len(points) == 2:
                blk = np.zeros(img.shape, np.uint8)  
                cv2.rectangle(blk, points[0], points[1], (153,135,255), -1)
                img = cv2.addWeighted(img, 1.0, blk, 0.5, 1)
                points = []
        return img if isinstance(img, np.ndarray) else np.array(img)
    input_image.select(
        get_point,
        [input_image, selected_points],
        [input_image],
    )

    def store_fg_img(img, original_image, selected_points):
        image = img
        image = Image.fromarray(image)
        image = image.resize((512,512), PIL.Image.BILINEAR)
        comp = original_image.copy()
        x1, y1 = selected_points[0]
        x2, y2 = selected_points[1]
        image_crop = image.resize((x2-x1, y2-y1), PIL.Image.BILINEAR)
        image = np.array(image)
        image_crop = np.array(image_crop)
        comp[y1:y2, x1:x2] = image_crop
        return image, comp
    fg_input_image.upload(
        store_fg_img,
        [fg_input_image, original_image, selected_points],
        [fg_input_image, comp_image]
    )

    def undo_points(original_image):
        return original_image, []
    undo_button.click(
        undo_points,
        [original_image],
        [input_image, selected_points]
    )

    run_button.click(
        inference,
        [original_image,
        input_image,
        fg_input_image,
        selected_points,
        ],
        [score]
    )

demo.queue().launch(share=True, debug=True)
