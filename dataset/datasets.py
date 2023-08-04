import sys,os
PROJ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_dir)
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import torch
import json
import random
from PIL import PngImagePlugin
import math

MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

IMAGE_NET_MEAN = [0.5, 0.5, 0.5]
IMAGE_NET_STD  = [0.5, 0.5, 0.5]

anno_root  = os.path.join(PROJ_dir, "data")
assert os.path.exists(anno_root), anno_root
cate_json  = os.path.join(anno_root, 'train/labels/category.json')
cate_data  = json.load(open(cate_json, 'r'))
ClassName  = list(cate_data.keys())
# make sure the category has fixed class index.
ClassName.sort()

def padding_to_specified_aspectratio(src_img, tar_w, tar_h, pad_pixel):
    src_h,src_w = src_img.shape[:2]
    h_scale = float(tar_h) / src_h
    w_scale = float(tar_w) / src_w
    if w_scale < h_scale:
        pad_w = 0
        pad_h = src_w * (float(tar_h) / tar_w) - src_h
    else:
        pad_h = 0
        pad_w = src_h * (float(tar_w) / tar_h) - src_w
    pad_w, pad_h = max(0, pad_w), max(0, pad_h)
    pad_y1 = int(pad_h // 2)
    pad_y2 = int(pad_h - pad_y1)
    pad_x1 = int(pad_w // 2)
    pad_x2 = int(pad_w - pad_x1)
    try:
        if len(src_img.shape) == 3:
            pad_im = np.pad(src_img, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0,0)),
                            'constant', constant_values=pad_pixel)
        else:
            pad_im = np.pad(src_img, ((pad_y1, pad_y2), (pad_x1, pad_x2)),
                            'constant', constant_values=pad_pixel)
    except:
        print(pad_x1, pad_y1, pad_x2, pad_y2, pad_w, pad_h)
    return pad_im


def padding_to_square(src_img, pad_pixel=255):
    src_h,src_w = src_img.shape[:2]
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

def normbox2globalbox(query_box, width, height):
    x1, y1, x2, y2 = query_box
    x1 = max(int(x1 * width), 0)
    y1 = max(int(y1 * height), 0)
    x2 = min(int(x2 * width), width)
    y2 = min(int(y2 * height), height)
    return [x1, y1, x2, y2]

def fill_box_with_specified_pixel(bg_im, query_box, fill_value):
    # filling the query box with specified value
    bg_w,bg_h = bg_im.size
    x1,y1,x2,y2 = normbox2globalbox(query_box, bg_w, bg_h)
    bg_im = np.asarray(bg_im)
    bg_im[y1:y2, x1:x2] = fill_value
    bg_im = Image.fromarray(bg_im)
    return bg_im

def random_padding_box(norm_box, max_pad):
    x1,y1,x2,y2 = norm_box
    w_ratio = random.uniform(0.05, max_pad)
    h_ratio = random.uniform(0.05, max_pad)
    pad_w = w_ratio * (x2 - x1)
    pad_h = h_ratio * (y2 - y1)
    x1 = max(0, x1 - pad_w / 2)
    y1 = max(0, y1 - pad_h / 2)
    x2 = min(1, x2 + pad_w / 2)
    y2 = min(1, y2 + pad_h / 2)
    return [x1,y1,x2,y2]

class TrainDataset(Dataset):
    def __init__(self, cfg, split='train'):
        assert split == 'train', split

        self.padding_max = cfg.padding_max
        self.img_aug = cfg.img_aug

        self.num_pos = cfg.num_positive
        self.num_neg = cfg.num_negative
        self.num_classes = cfg.num_classes
        self.fill_pixel = cfg.fill_pixel
        self.pad_pixel = cfg.pad_pixel
        self.data_root = os.path.join(anno_root, split)
        self.bg_folder = os.path.join(self.data_root, 'bg')
        self.fg_folder = os.path.join(self.data_root, 'fg')
        self.anno_file = os.path.join(self.data_root, 'labels', f'{split}_{cfg.config_name}.json')
        assert os.path.exists(self.anno_file), self.anno_file
        with open(self.anno_file, 'r') as f:
            self.anno = json.load(f)

        if self.num_classes < len(ClassName):
            image_list = []
            for cate in ClassName[:self.num_classes]:
                image_list += cate_data[cate]['images']
            self.image_list = image_list
        else:
            self.image_list = list(self.anno.keys())

        self.image_size = (cfg.image_size, cfg.image_size)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
        self.transformer_withresize = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.image_list)

    def _get_random_aug(self):
        neg_deg = np.random.randint(90, 180)
        neg_shear = np.random.randint(15, 60)
        aug0_b = np.random.uniform(0.5, 1.5)
        aug0_c = np.random.uniform(0.5, 1.5)
        aug0_s = np.random.uniform(0.5, 1.5)
        aug0_h = np.random.uniform(-0.5, 0.5)
        aug1_ker = np.random.choice(list(range(11, 31, 2)))
        aug1_sigma= np.random.randint(10, 20)

        self.transformer_withresize_pos_aug0 = transforms.Compose([
            transforms.ColorJitter(brightness=(aug0_b, aug0_b), contrast=(aug0_c, aug0_c), saturation=(aug0_s, aug0_s), hue=(aug0_h, aug0_h)),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
        self.transformer_withresize_pos_aug1 = transforms.Compose([
            transforms.GaussianBlur(aug1_ker, aug1_sigma),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
        self.transformer_withresize_neg_aug = transforms.Compose([
            transforms.RandomAffine(degrees=(neg_deg, neg_deg), translate=None, scale=None, shear=(neg_shear, neg_shear), resample=0, fillcolor=(255, 255, 255)),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])


        self.transformer_withresize_pos_aug0_for_comp = transforms.Compose([
            transforms.ColorJitter(brightness=(aug0_b, aug0_b), contrast=(aug0_c, aug0_c), saturation=(aug0_s, aug0_s), hue=(aug0_h, aug0_h)),
        ])
        self.transformer_withresize_pos_aug1_for_comp = transforms.Compose([
            transforms.GaussianBlur(aug1_ker, aug1_sigma),
        ])
        self.transformer_withresize_neg_aug_for_comp = transforms.Compose([
            transforms.RandomAffine(degrees=(neg_deg, neg_deg), translate=None, scale=None, shear=(neg_shear, neg_shear), resample=0, fillcolor=(255, 255, 255)),
        ])

    def __getitem__(self, index):
        bg_name = self.image_list[index]
        bg_file = os.path.join(self.bg_folder, bg_name)
        assert os.path.exists(bg_file), bg_file
        bg_im = Image.open(bg_file).convert("RGB")
        norm_box = self.anno[bg_name]['bbox']

        self._get_random_aug()

        if self.padding_max > 0:
            norm_box = random_padding_box(norm_box, self.padding_max)

        bg_im = fill_box_with_specified_pixel(bg_im, norm_box, self.fill_pixel)
        bg_t = self.transformer_withresize(bg_im)
        bg_ori_w, bg_ori_h = bg_im.size

        rs_bg = bg_im.resize(self.image_size)
        bg_im = np.asarray(bg_im)

        ori_x1, ori_y1, ori_x2, ori_y2 = normbox2globalbox(norm_box, bg_ori_w, bg_ori_h)
        ori_tar_w = ori_x2 - ori_x1
        ori_tar_h = ori_y2 - ori_y1

        rs_bg = np.asarray(rs_bg)
        query_box  = normbox2globalbox(norm_box, rs_bg.shape[1], rs_bg.shape[0])
        x1, y1, x2, y2 = query_box
        tar_w = x2 - x1
        tar_h = y2 - y1

        add_w = int(ori_tar_w * (math.sqrt(2) - 1) / 2)
        add_h = int(ori_tar_h * (math.sqrt(2) - 1) / 2)
        new_y1 = max(0, ori_y1 - add_h)
        new_y2 = min(bg_ori_h, ori_y2 + add_h)
        new_x1 = max(0, ori_x1 - add_w)
        new_x2 = min(bg_ori_w, ori_x2 + add_w)
        new_normbox = [float(new_x1) / bg_ori_w, float(new_y1) / bg_ori_h,
                       float(new_x2) / bg_ori_w, float(new_y2) / bg_ori_h]
        new_box = normbox2globalbox(new_normbox, rs_bg.shape[1], rs_bg.shape[0])

        sim_imgs = self.anno[bg_name]['similars']
        sim_tmp = [bg_name]
        if self.num_pos > 1:
            random.shuffle(sim_imgs)
            for i in range(self.num_pos - 1):
                sim_tmp.append(sim_imgs[i % len(sim_imgs)])
        sim_imgs = sim_tmp

        dis_imgs = self.anno[bg_name]['dissimilars']
        if self.num_neg > 1:
            random.shuffle(dis_imgs)
        tmp_dis = []
        for i in range(self.num_neg):
            tmp_dis.append(dis_imgs[i % len(dis_imgs)])
        dis_imgs = tmp_dis

        pos_fgs, pos_comps, pos_scale_comps, pos_files = [], [], [], []
        neg_fgs, neg_comps, neg_scale_comps, neg_files = [], [], [],[]
        cnt_pos = 0
        for im_name in sim_imgs:
            flag_pos = 0
            fg_file = os.path.join(self.fg_folder, im_name)
            fg = Image.open(fg_file).convert('RGB')
            fg = np.asarray(fg)

            rs_comp = rs_bg.copy()
            rs_fg   = cv2.resize(fg.copy(), (tar_w, tar_h))
                
            rs_comp[y1:y2,x1:x2] = rs_fg
            rs_comp = Image.fromarray(rs_comp)
            rs_comp_t = self.transformer(rs_comp)
            pos_comps.append(rs_comp_t)

            if im_name.split(".")[0] == bg_name.split(".")[0] and self.img_aug == True and cnt_pos == 0:
                flag_pos = 1

                rs_pad_fg = Image.fromarray(fg.copy())
                rs_pad_fg_aug0 = np.asarray(self.transformer_withresize_pos_aug0_for_comp(rs_pad_fg))
                rs_pad_fg_aug0 = cv2.resize(rs_pad_fg_aug0, ((tar_w, tar_h)))
                rs_comp_aug0 = rs_bg.copy()
                rs_comp_aug0[y1:y2,x1:x2] = rs_pad_fg_aug0
                rs_comp_aug0 = Image.fromarray(rs_comp_aug0)
                rs_comp_aug0_t = self.transformer(rs_comp_aug0)
                pos_comps.append(rs_comp_aug0_t)

                rs_pad_fg = Image.fromarray(fg.copy())
                rs_pad_fg_aug1 = np.asarray(self.transformer_withresize_pos_aug1_for_comp(rs_pad_fg))
                rs_pad_fg_aug1 = cv2.resize(rs_pad_fg_aug1, ((tar_w, tar_h)))
                rs_comp_aug1 = rs_bg.copy()
                rs_comp_aug1[y1:y2,x1:x2] = rs_pad_fg_aug1
                rs_comp_aug1 = Image.fromarray(rs_comp_aug1)
                rs_comp_aug1_t = self.transformer(rs_comp_aug1)
                pos_comps.append(rs_comp_aug1_t)

                rs_pad_fg = Image.fromarray(fg.copy())
                rs_pad_fg_neg = np.asarray(self.transformer_withresize_neg_aug_for_comp(rs_pad_fg))
                rs_pad_fg_neg = cv2.resize(rs_pad_fg_neg, ((tar_w, tar_h)))
                rs_comp_neg = rs_bg.copy()
                rs_comp_neg[y1:y2,x1:x2] = rs_pad_fg_neg
                rs_comp_neg = Image.fromarray(rs_comp_neg)
                rs_comp_neg_t = self.transformer(rs_comp_neg)
                neg_comps.append(rs_comp_neg_t)

            ori_fg   = cv2.resize(fg.copy(), (ori_tar_w, ori_tar_h))
            ori_comp = bg_im.copy()
            ori_comp[ori_y1:ori_y2, ori_x1:ori_x2] = ori_fg
            scale_comp = ori_comp[new_y1:new_y2, new_x1:new_x2].copy()
            scale_comp_r = Image.fromarray(scale_comp).resize(self.image_size)
            pos_scale_comps.append(self.transformer(scale_comp_r))

            if im_name.split(".")[0] == bg_name.split(".")[0] and self.img_aug == True and cnt_pos == 0:

                ori_pad_fg = Image.fromarray(fg.copy())
                ori_pad_fg_aug0 = np.asarray(self.transformer_withresize_pos_aug0_for_comp(ori_pad_fg))
                ori_pad_fg_aug0 = cv2.resize(ori_pad_fg_aug0, ((ori_tar_w, ori_tar_h)))
                ori_comp_aug0 = bg_im.copy()
                ori_comp_aug0[ori_y1:ori_y2, ori_x1:ori_x2] = ori_pad_fg_aug0
                ori_scale_aug0 = ori_comp_aug0[new_y1:new_y2, new_x1:new_x2].copy()
                ori_scale_aug0 = Image.fromarray(ori_scale_aug0).resize(self.image_size)
                ori_scale_aug0_t = self.transformer(ori_scale_aug0)
                pos_scale_comps.append(ori_scale_aug0_t)

                ori_pad_fg = Image.fromarray(fg.copy())
                ori_pad_fg_aug1 = np.asarray(self.transformer_withresize_pos_aug1_for_comp(ori_pad_fg))
                ori_pad_fg_aug1 = cv2.resize(ori_pad_fg_aug1, ((ori_tar_w, ori_tar_h)))
                ori_comp_aug1 = bg_im.copy()
                ori_comp_aug1[ori_y1:ori_y2, ori_x1:ori_x2] = ori_pad_fg_aug1
                ori_scale_aug1 = ori_comp_aug1[new_y1:new_y2, new_x1:new_x2].copy()
                ori_scale_aug1 = Image.fromarray(ori_scale_aug1).resize(self.image_size)
                ori_scale_aug1_t = self.transformer(ori_scale_aug1)
                pos_scale_comps.append(ori_scale_aug1_t)

                ori_pad_fg = Image.fromarray(fg.copy())
                ori_pad_fg_neg = np.asarray(self.transformer_withresize_neg_aug_for_comp(ori_pad_fg))
                ori_pad_fg_neg = cv2.resize(ori_pad_fg_neg, ((ori_tar_w, ori_tar_h)))
                ori_comp_neg = bg_im.copy()
                ori_comp_neg[ori_y1:ori_y2, ori_x1:ori_x2] = ori_pad_fg_neg
                ori_scale_neg = ori_comp_neg[new_y1:new_y2, new_x1:new_x2].copy()
                ori_scale_neg = Image.fromarray(ori_scale_neg).resize(self.image_size)
                ori_scale_neg_t = self.transformer(ori_scale_neg)
                neg_scale_comps.append(ori_scale_neg_t)

            pad_fg = padding_to_square(fg.copy(), self.pad_pixel)
            pad_fg = Image.fromarray(pad_fg)

            if im_name.split(".")[0] == bg_name.split(".")[0] and self.img_aug == True and cnt_pos == 0:
                pos_fgs.append(self.transformer_withresize_pos_aug0(pad_fg))
                pos_fgs.append(self.transformer_withresize_pos_aug1(pad_fg))
                neg_fgs.append(self.transformer_withresize_neg_aug(pad_fg))

            pos_fgs.append(self.transformer_withresize(pad_fg))
            pos_files.append(fg_file)

            if flag_pos:
                cnt_pos += 1

        pos_fgs   = torch.stack(pos_fgs, dim=0)
        pos_comps = torch.stack(pos_comps, dim=0)
        pos_scale_comps = torch.stack(pos_scale_comps, dim=0)

        for im_name in dis_imgs:
            fg_file = os.path.join(self.fg_folder, im_name)
            fg = Image.open(fg_file).convert('RGB')
            fg = np.asarray(fg)

            rs_comp = rs_bg.copy()
            rs_fg = cv2.resize(fg.copy(), (tar_w, tar_h))
            rs_comp[y1:y2, x1:x2] = rs_fg
            rs_comp_t = self.transformer(Image.fromarray(rs_comp))
            neg_comps.append(rs_comp_t)

            ori_fg = cv2.resize(fg.copy(), (ori_tar_w, ori_tar_h))
            ori_comp = bg_im.copy()
            ori_comp[ori_y1:ori_y2, ori_x1:ori_x2] = ori_fg
            scale_comp = ori_comp[new_y1:new_y2, new_x1:new_x2].copy()
            scale_comp_r = Image.fromarray(scale_comp).resize(self.image_size)
            neg_scale_comps.append(self.transformer(scale_comp_r))

            pad_fg = padding_to_square(fg.copy(), self.pad_pixel)
            pad_fg = Image.fromarray(pad_fg)

            neg_fgs.append(self.transformer_withresize(pad_fg))
            neg_files.append(fg_file)

        neg_fgs = torch.stack(neg_fgs, dim=0)
        neg_comps = torch.stack(neg_comps, dim=0)
        neg_scale_comps = torch.stack(neg_scale_comps, dim=0)

        cls_name = self.anno[bg_name]['class']
        cls_idx = ClassName.index(cls_name)

        sample = {
            'bg': bg_t,
            'pos_fg': pos_fgs,
            'neg_fg': neg_fgs,
            'query_box': torch.tensor(query_box).float(),
            'crop_box': torch.tensor(new_box).float(),
            'pos_comp': pos_comps,
            'pos_scale_comp':pos_scale_comps,
            'neg_comp': neg_comps,
            'neg_scale_comp':neg_scale_comps,
            'bg_file': bg_name,
            'pos_file': sim_imgs,
            'neg_file': dis_imgs,
            'cls_idx': cls_idx,
            'cls': cls_name,
        }
        return sample

class TestDataset(Dataset):
    def __init__(self, cfg, category, bg_index, split='test'):
        super().__init__()
        self.category = category
        self.bg_index = bg_index
        assert split == 'test', split
        self.fill_pixel = cfg.fill_pixel
        self.pad_pixel  = cfg.pad_pixel

        self.data_root = os.path.join(anno_root, split)
        self.bg_folder = os.path.join(self.data_root, 'bg')
        self.fg_folder = os.path.join(self.data_root, 'fg')
        self.anno_file = os.path.join(self.data_root, 'labels', f'{split}.json')
        assert os.path.exists(self.anno_file), self.anno_file
        with open(self.anno_file, 'r') as f:
            self.anno = json.load(f)
        self.image_list = list(self.anno[category]['bg'])
        self.image_list.sort()
        assert self.bg_index < len(self.image_list), \
            f'The index({self.bg_index}) cannot exceed the list size({len(self.image_list)}).'
        self.bg_name = self.image_list[self.bg_index][0]
        self.fgs = list(self.anno[category]['fg'])

        self.image_size = (cfg.image_size, cfg.image_size)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
        self.transformer_withresize = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.fgs)

    def __getitem__(self, index):
        bg_name = self.bg_name
        bg_file = os.path.join(self.bg_folder, bg_name)
        assert os.path.exists(bg_file), bg_file
        bg_im = Image.open(bg_file).convert("RGB")
        norm_box = self.image_list[self.bg_index][1]
        bg_im = fill_box_with_specified_pixel(bg_im, norm_box, self.fill_pixel)
        bg_t = self.transformer_withresize(bg_im)

        bg_ori_w, bg_ori_h = bg_im.size
        rs_bg = bg_im.resize(self.image_size)
        bg_im = np.asarray(bg_im)

        ori_x1, ori_y1, ori_x2, ori_y2 = normbox2globalbox(norm_box, bg_ori_w, bg_ori_h)
        ori_tar_w = ori_x2 - ori_x1
        ori_tar_h = ori_y2 - ori_y1
        
        bg_ratio = ori_tar_w / ori_tar_h

        rs_bg = np.asarray(rs_bg)
        query_box = normbox2globalbox(norm_box, rs_bg.shape[1], rs_bg.shape[0])
        x1, y1, x2, y2 = query_box
        tar_w = x2 - x1
        tar_h = y2 - y1

        add_w = int(ori_tar_w * (math.sqrt(2) - 1) / 2)
        add_h = int(ori_tar_h * (math.sqrt(2) - 1) / 2)
        new_y1 = max(0, ori_y1 - add_h)
        new_y2 = min(bg_ori_h, ori_y2 + add_h)
        new_x1 = max(0, ori_x1 - add_w)
        new_x2 = min(bg_ori_w, ori_x2 + add_w)
        new_normbox = [float(new_x1) / bg_ori_w, float(new_y1) / bg_ori_h,
                       float(new_x2) / bg_ori_w, float(new_y2) / bg_ori_h]
        new_box = normbox2globalbox(new_normbox, rs_bg.shape[1], rs_bg.shape[0])

        fg_name = self.fgs[index]
        fg_file = os.path.join(self.fg_folder, fg_name)
        fg = Image.open(fg_file).convert('RGB')
        fg_w, fg_h = fg.size
        fg_ratio = fg_w / fg_h
        ratio_dif = abs(bg_ratio - fg_ratio)
        fg = np.asarray(fg)

        rs_comp = rs_bg.copy()
        rs_fg   = cv2.resize(fg.copy(), (tar_w, tar_h))
        rs_comp[y1:y2, x1:x2] = rs_fg
        rs_comp_t = self.transformer(Image.fromarray(rs_comp))

        ori_fg = cv2.resize(fg.copy(), (ori_tar_w, ori_tar_h))
        ori_comp = bg_im.copy()
        ori_comp[ori_y1:ori_y2, ori_x1:ori_x2] = ori_fg
        scale_comp = ori_comp[new_y1:new_y2, new_x1:new_x2].copy()
        scale_comp_r = Image.fromarray(scale_comp).resize(self.image_size)
        scale_comp_t = self.transformer(scale_comp_r)

        pad_fg = padding_to_square(fg.copy(), self.pad_pixel)
        pad_fg = Image.fromarray(pad_fg)
        fg_t = self.transformer_withresize(pad_fg)

        cls_name = self.category
        cls_idx = ClassName.index(cls_name)

        sample = {
            'bg': bg_t,
            'fg': fg_t,
            'bg_file': bg_file,
            'fg_name': fg_name,
            'query_box': torch.tensor(query_box).float(),
            'crop_box': torch.tensor(new_box).float(),
            'comp': rs_comp_t,
            'scale_comp': scale_comp_t,
            'cls_idx': cls_idx,
            'cls': cls_name,
            'ratio_dif': ratio_dif,
        }
        return sample

class Test2Dataset(Dataset):
    def __init__(self, cfg, category, bg_index, split='test'):
        super().__init__()
        self.category = category
        self.bg_index = bg_index
        assert split == 'test', split
        self.fill_pixel = cfg.fill_pixel
        self.pad_pixel  = cfg.pad_pixel

        self.data_root = os.path.join(anno_root, split)
        self.bg_folder = os.path.join(self.data_root, 'bg_set2')
        self.fg_folder = os.path.join(self.data_root, 'fg')
        self.anno_file = os.path.join(self.data_root, 'labels', f'test_set2.json')
    
        assert os.path.exists(self.anno_file), self.anno_file
        with open(self.anno_file, 'r') as f:
            self.anno = json.load(f)

        self.bg_list = list(self.anno[category].keys())
        self.bg_list.sort()
        assert self.bg_index < len(self.bg_list), \
            f'The index({self.bg_index}) cannot exceed the list size({len(self.bg_list)}).'
        self.bg_name = self.bg_list[self.bg_index]
        self.bg_anno = self.anno[category][self.bg_name]
        self.fgs = list(self.bg_anno[-1].keys())
        self.fgs.sort()

        self.image_size = (cfg.image_size, cfg.image_size)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
        self.transformer_withresize = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.fgs)

    def __getitem__(self, index):
        bg_name = self.bg_name
        bg_file = os.path.join(self.bg_folder, bg_name)
        assert os.path.exists(bg_file), bg_file
        bg_im = Image.open(bg_file).convert("RGB")
        norm_box = self.bg_anno[0]

        bg_im = fill_box_with_specified_pixel(bg_im, norm_box, self.fill_pixel)
        bg_t = self.transformer_withresize(bg_im)

        bg_ori_w, bg_ori_h = bg_im.size
        rs_bg = bg_im.resize(self.image_size)
        bg_im = np.asarray(bg_im)

        ori_x1, ori_y1, ori_x2, ori_y2 = normbox2globalbox(norm_box, bg_ori_w, bg_ori_h)
        ori_tar_w = ori_x2 - ori_x1
        ori_tar_h = ori_y2 - ori_y1
        
        bg_ratio = ori_tar_w / ori_tar_h

        rs_bg = np.asarray(rs_bg)
        query_box = normbox2globalbox(norm_box, rs_bg.shape[1], rs_bg.shape[0])
        x1, y1, x2, y2 = query_box
        tar_w = x2 - x1
        tar_h = y2 - y1

        add_w = int(ori_tar_w * (math.sqrt(2) - 1) / 2)
        add_h = int(ori_tar_h * (math.sqrt(2) - 1) / 2)
        new_y1 = max(0, ori_y1 - add_h)
        new_y2 = min(bg_ori_h, ori_y2 + add_h)
        new_x1 = max(0, ori_x1 - add_w)
        new_x2 = min(bg_ori_w, ori_x2 + add_w)
        new_normbox = [float(new_x1) / bg_ori_w, float(new_y1) / bg_ori_h,
                       float(new_x2) / bg_ori_w, float(new_y2) / bg_ori_h]
        new_box = normbox2globalbox(new_normbox, rs_bg.shape[1], rs_bg.shape[0])

        fg_name = self.fgs[index]
        fg_label= self.bg_anno[-1][fg_name]
        fg_file = os.path.join(self.fg_folder, fg_name)
        fg = Image.open(fg_file).convert('RGB')
        fg_w, fg_h = fg.size
        fg_ratio = fg_w / fg_h
        ratio_dif = abs(bg_ratio - fg_ratio)
        fg = np.asarray(fg)

        rs_comp = rs_bg.copy()
        rs_fg   = cv2.resize(fg.copy(), (tar_w, tar_h))
        rs_comp[y1:y2, x1:x2] = rs_fg
        rs_comp_t = self.transformer(Image.fromarray(rs_comp))

        ori_fg = cv2.resize(fg.copy(), (ori_tar_w, ori_tar_h))
        ori_comp = bg_im.copy()
        ori_comp[ori_y1:ori_y2, ori_x1:ori_x2] = ori_fg
        scale_comp = ori_comp[new_y1:new_y2, new_x1:new_x2].copy()
        scale_comp_r = Image.fromarray(scale_comp).resize(self.image_size)
        scale_comp_t = self.transformer(scale_comp_r)

        pad_fg = padding_to_square(fg.copy(), self.pad_pixel)
        pad_fg = Image.fromarray(pad_fg)
        fg_t = self.transformer_withresize(pad_fg)

        cls_name = self.category
        cls_idx = ClassName.index(cls_name)

        sample = {
            'bg': bg_t,
            'fg': fg_t,
            'label': fg_label,
            'bg_file': bg_file,
            'fg_file': fg_file,
            'query_box': torch.tensor(query_box).float(),
            'crop_box': torch.tensor(new_box).float(),
            'comp': rs_comp_t,
            'scale_comp': scale_comp_t,
            'cls_idx': cls_idx,
            'cls': cls_name,
            'ratio_dif': ratio_dif,
        }
        return sample
        
class Test2DatasetUnconstrained(Dataset):
    def __init__(self, cfg, category, bg_index, split='test'):
        super().__init__()
        self.category = category
        self.bg_index = bg_index
        assert split == 'test', split
        self.fill_pixel = cfg.fill_pixel
        self.pad_pixel  = cfg.pad_pixel

        self.data_root = os.path.join(anno_root, split)
        self.bg_folder = os.path.join(self.data_root, 'bg_set2')
        self.fg_folder = os.path.join(self.data_root, 'fg')
        self.anno_file = os.path.join(self.data_root, 'labels', f'test_set2.json')
        assert os.path.exists(self.anno_file), self.anno_file
        with open(self.anno_file, 'r') as f:
            self.anno = json.load(f)

        self.bg_list = list(self.anno[category].keys())
        self.bg_list.sort()
        assert self.bg_index < len(self.bg_list), \
            f'The index({self.bg_index}) cannot exceed the list size({len(self.bg_list)}).'
        self.bg_name = self.bg_list[self.bg_index]
        self.bg_anno = self.anno[category][self.bg_name]
        self.fgs = list(os.listdir(self.fg_folder))
        self.fgs.sort()

        self.image_size = (cfg.image_size, cfg.image_size)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
        self.transformer_withresize = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.fgs)

    def __getitem__(self, index):
        bg_name = self.bg_name
        bg_file = os.path.join(self.bg_folder, bg_name)
        assert os.path.exists(bg_file), bg_file
        bg_im = Image.open(bg_file).convert("RGB")
        norm_box = self.bg_anno[0]

        bg_im = fill_box_with_specified_pixel(bg_im, norm_box, self.fill_pixel)
        bg_t = self.transformer_withresize(bg_im)

        bg_ori_w, bg_ori_h = bg_im.size
        rs_bg = bg_im.resize(self.image_size)
        bg_im = np.asarray(bg_im)

        ori_x1, ori_y1, ori_x2, ori_y2 = normbox2globalbox(norm_box, bg_ori_w, bg_ori_h)
        ori_tar_w = ori_x2 - ori_x1
        ori_tar_h = ori_y2 - ori_y1
        
        bg_ratio = ori_tar_w / ori_tar_h

        rs_bg = np.asarray(rs_bg)
        query_box = normbox2globalbox(norm_box, rs_bg.shape[1], rs_bg.shape[0])
        x1, y1, x2, y2 = query_box
        tar_w = x2 - x1
        tar_h = y2 - y1

        add_w = int(ori_tar_w * (math.sqrt(2) - 1) / 2)
        add_h = int(ori_tar_h * (math.sqrt(2) - 1) / 2)
        new_y1 = max(0, ori_y1 - add_h)
        new_y2 = min(bg_ori_h, ori_y2 + add_h)
        new_x1 = max(0, ori_x1 - add_w)
        new_x2 = min(bg_ori_w, ori_x2 + add_w)
        new_normbox = [float(new_x1) / bg_ori_w, float(new_y1) / bg_ori_h,
                       float(new_x2) / bg_ori_w, float(new_y2) / bg_ori_h]
        new_box = normbox2globalbox(new_normbox, rs_bg.shape[1], rs_bg.shape[0])

        fg_name = self.fgs[index]
        fg_file = os.path.join(self.fg_folder, fg_name)
        fg = Image.open(fg_file).convert('RGB')
        fg_w, fg_h = fg.size
        fg_ratio = fg_w / fg_h
        ratio_dif = abs(bg_ratio - fg_ratio)
        fg = np.asarray(fg)

        rs_comp = rs_bg.copy()
        rs_fg   = cv2.resize(fg.copy(), (tar_w, tar_h))
        rs_comp[y1:y2, x1:x2] = rs_fg
        rs_comp_t = self.transformer(Image.fromarray(rs_comp))

        ori_fg = cv2.resize(fg.copy(), (ori_tar_w, ori_tar_h))
        ori_comp = bg_im.copy()
        ori_comp[ori_y1:ori_y2, ori_x1:ori_x2] = ori_fg
        scale_comp = ori_comp[new_y1:new_y2, new_x1:new_x2].copy()
        scale_comp_r = Image.fromarray(scale_comp).resize(self.image_size)
        scale_comp_t = self.transformer(scale_comp_r)

        pad_fg = padding_to_square(fg.copy(), self.pad_pixel)
        pad_fg = Image.fromarray(pad_fg)
        fg_t = self.transformer_withresize(pad_fg)

        cls_name = self.category
        cls_idx = ClassName.index(cls_name)

        sample = {
            'bg': bg_t,
            'fg': fg_t,
            'bg_file': bg_file,
            'fg_file': fg_file,
            'query_box': torch.tensor(query_box).float(),
            'crop_box': torch.tensor(new_box).float(),
            'comp': rs_comp_t,
            'scale_comp': scale_comp_t,
            'cls_idx': cls_idx,
            'cls': cls_name,
            'ratio_dif': ratio_dif,
        }
        return sample

def dataset_count():
    train_json = os.path.join(anno_root, 'train', 'labels', 'train.json')
    with open(train_json, 'r') as f:
        train_anno = json.load(f)
    train_bg = list(train_anno.keys())
    train_fg = list(train_anno.keys())
    print(f'training set: {len(train_bg)} bg, {len(train_fg)} fg')

    test1_json = os.path.join(anno_root, 'test', 'labels', 'test.json')
    with open(test1_json, 'r') as f:
        test1_anno = json.load(f)
    test1_bg, test1_fg = [], []
    for c in test1_anno.keys():
        test1_fg.extend(test1_anno[c]['fg'])
        for e in test1_anno[c]['bg']:
            test1_bg.append(e[0])
    print(f'test set1: {len(test1_bg)} bg, {len(test1_fg)} fg')

    test2_json = os.path.join(anno_root, 'test', 'labels', 'test_set2.json')
    with open(test2_json, 'r') as f:
        test2_anno = json.load(f)
    test2_bg, test2_fg = [], []
    for c in test2_anno.keys():
        bg_list = list(test2_anno[c].keys())
        test2_bg.extend(bg_list)
        fg_list = list(test2_anno[c][bg_list[0]][-1].keys())
        test2_fg.extend(fg_list)
    print(f'test set2: {len(test2_bg)} bg, {len(test2_fg)} fg')

    unique_bg, unique_fg = set(train_bg + test1_bg + test2_bg), set(train_fg + test1_fg + test2_fg)
    print(f'total {len(unique_bg)} bg, {len(unique_fg)} fg')