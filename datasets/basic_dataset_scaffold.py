from joblib import Parallel, delayed
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import tqdm


def pil_ensure_3dim(img):
    if len(img.size) == 2:
        img = img.convert('RGB')
    return img


class BaseDataset(Dataset):
    def __init__(self, image_dict, opt, is_validation=False):
        self.pars = opt
        self.is_validation = is_validation
        self.image_dict = image_dict

        self.init_setup()

        if 'bninception' not in opt.arch:
            if 'clipi' in opt.arch:
                norm_data = {
                    'mean': [0.48145466, 0.4578275, 0.40821073],
                    'std': [0.26862954, 0.26130258, 0.27577711]
                }
                # self.f_norm = normalize = A.Normalize(**norm_data, always_apply=True)
                self.f_tensor_norm = transforms.Normalize(**norm_data)
            else:
                norm_data = {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
                # self.f_norm = normalize = A.Normalize(**norm_data, always_apply=True)
                self.f_tensor_norm = transforms.Normalize(**norm_data)
        else:
            # normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[1., 1., 1.])
            norm_data = {
                'mean': [0.502, 0.4588, 0.4078],
                'std': [0.0039, 0.0039, 0.0039]
            }
            # self.f_norm = normalize = A.Normalize(**norm_data, always_apply=True)
            self.f_tensor_norm = transforms.Normalize(**norm_data)

        self.crop_size = crop_im_size = [
            224, 224
        ] if 'googlenet' not in opt.arch else [227, 227]
        self.base_size = 256
        if opt.augmentation == 'big': self.crop_size = [256, 256]
        self.provide_transforms()

    def provide_transforms(self):
        self.normal_transform = []
        if not self.is_validation:
            if self.pars.augmentation == 'base' or self.pars.augmentation == 'big':
                self.normal_transform.extend([
                    transforms.RandomResizedCrop(size=self.crop_size[0]),
                    transforms.RandomHorizontalFlip(0.5)
                ])
            elif self.pars.augmentation == 'adv':
                self.normal_transform.extend([
                    transforms.RandomResizedCrop(size=self.crop_size[0]),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomHorizontalFlip(0.5)
                ])
            elif self.pars.augmentation == 'red':
                self.normal_transform.extend([
                    transforms.Resize(size=256),
                    transforms.RandomCrop(self.crop_size[0]),
                    transforms.RandomHorizontalFlip(0.5)
                ])
        else:
            self.normal_transform.extend([
                transforms.Resize(256),
                transforms.CenterCrop(self.crop_size[0])
            ])

        self.normal_transform = transforms.Compose(self.normal_transform)
        self.shared_transform = transforms.Compose(
            [transforms.ToTensor(), self.f_tensor_norm])

    def init_setup(self):
        self.n_files = np.sum(
            [len(self.image_dict[key]) for key in self.image_dict.keys()])
        self.avail_classes = sorted(list(self.image_dict.keys()))

        counter = 0
        temp_image_dict = {}
        for i, key in enumerate(self.avail_classes):
            temp_image_dict[key] = []
            for path in self.image_dict[key]:
                temp_image_dict[key].append([path, counter])
                counter += 1

        self.image_dict = temp_image_dict
        self.image_list = [[(x[0], key) for x in self.image_dict[key]]
                           for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        self.image_paths = self.image_list

        self.is_init = True

    def __getitem__(self, idx):
        image_path = self.image_list[idx][0]
        input_image = pil_ensure_3dim(Image.open(image_path))
        imrot_class = -1

        out_dict = {}
        out_dict['image'] = self.normal_transform(input_image)
        out_dict['image'] = self.shared_transform(out_dict['image'])

        if 'bninception' in self.pars.arch:
            out_dict['image'] = out_dict['image'][range(3)[::-1], :]
            if self.aux_preprocess is not None:
                out_dict['aux_prep_image'] = out_dict['aux_prep_image'][
                    range(3)[::-1], :]

        out_dict['path'] = image_path
        return self.image_list[idx][-1], out_dict, idx

    def __len__(self):
        return self.n_files
