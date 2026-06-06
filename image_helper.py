from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np
import torchvision.transforms as T

from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18
logger = logging.getLogger("logger")
import config
from config import device
import copy
import cv2

import yaml

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
import json
from trigger_family import TriggerConfig, apply_trigger_fragment


class ImageHelper(Helper):

    def _prefilter__ensure_cfg(self):
        """Ensure online prefilter parameters exist and have defaults."""

        self.client_filter = bool(self.params.get('client_filter', False))

        self.prefilter_threshold = float(self.params.get('prefilter_threshold', 0.5))

        self.prefilter_apply_to = str(self.params.get('prefilter_apply_to', 'poisoned')).lower()

        self.prefilter_path = self.params.get('prefilter_path', 'saved_models/prefilter/grid_best1.pth')

        self._prefilter_device = device

        if not hasattr(self, "_prefilter_model"):
            self._prefilter_model = None

    def _prefilter__load_model(self):
        """Lazily load the prefilter model on first use."""
        if getattr(self, "_prefilter_model", None) is not None:
            return
        from models.model_resnet_grid import ResNet18Dist

        assert isinstance(self.prefilter_path, str) and len(self.prefilter_path) > 0, \
            "[PREFILTER] prefilter_path is not set"


        ckpt = torch.load(self.prefilter_path, map_location='cpu')
        if isinstance(ckpt, dict):
            if isinstance(ckpt.get('state_dict', None), dict):
                state = ckpt['state_dict']
            elif isinstance(ckpt.get('model', None), dict):
                state = ckpt['model']
            else:
                state = ckpt
        else:
            state = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt


        def strip_prefix(d, prefix):
            if any(k.startswith(prefix) for k in d.keys()):
                return {k[len(prefix):]: v for k, v in d.items()}
            return d
        state = strip_prefix(state, 'module.')
        state = strip_prefix(state, 'model.')


        in_dim_ckpt = None
        for k, v in state.items():
            if k.endswith('dist_stem._proj.0.weight') and v.ndim == 1:
                in_dim_ckpt = int(v.numel())
                break


        bins, grid_size = 32, -1

        if in_dim_ckpt == 432:
            bins, grid_size = 12, 3

        elif in_dim_ckpt == 6912:
            bins, grid_size = 32, 8

        mdl = ResNet18Dist(pretrained=False, bins=bins, dist_dim=128, grid_size=grid_size)


        try:
            with torch.no_grad():
                if self.params.get('type') == config.TYPE_CIFAR:
                    _sz = 32
                elif self.params.get('type') == config.TYPE_TINYIMAGENET:
                    _sz = 64
                else:
                    _sz = int(self.params.get('prefilter_warmup_size', 224))
                _ = mdl.dist_stem(torch.zeros(1, 3, _sz, _sz))
        except Exception as e:
            print(f"[PREFILTER] warmup failed (non-fatal): {e}")


        model_dict = mdl.state_dict()
        compatible = {}
        missing_keys = []
        unexpected_keys = []
        for k, v in state.items():
            if k in model_dict:
                if tuple(v.shape) == tuple(model_dict[k].shape):
                    compatible[k] = v
                else:

                    pass
            else:
                unexpected_keys.append(k)

        model_dict.update(compatible)
        mdl.load_state_dict(model_dict, strict=False)


        for k in mdl.state_dict().keys():
            if k not in compatible and k not in state:
                missing_keys.append(k)

        if missing_keys:
            print(f"[PREFILTER] Warning: missing keys (not loaded): {missing_keys[:10]}{' ...' if len(missing_keys)>10 else ''}")
        if unexpected_keys:
            print(f"[PREFILTER] Warning: unexpected keys (ignored): {unexpected_keys[:10]}{' ...' if len(unexpected_keys)>10 else ''}")

        self._prefilter_model = mdl.to(self._prefilter_device).eval()

    def _prefilter__score_tensor_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Score an image tensor batch [B, C, H, W] and return suspicious probabilities [B].
        Used only by the online prefilter.
        """
        self._prefilter__ensure_cfg()
        self._prefilter__load_model()


        if imgs.device != self._prefilter_device:
            imgs = imgs.to(self._prefilter_device)

        with torch.no_grad():
            out = self._prefilter_model(imgs)

            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.ndim == 1:
                probs = torch.sigmoid(out)
            elif out.ndim == 2:
                if out.size(1) == 1:
                    probs = torch.sigmoid(out[:, 0])
                else:
                    probs = F.softmax(out, dim=1)[:, -1]
            else:
                probs = torch.sigmoid(out.view(out.size(0), -1).norm(dim=1))
        return probs.detach().float().cpu()

    def prefilter_batch_if_needed(self, data: torch.Tensor, targets: torch.Tensor, agent_name_key=None):
        """
        Filter one data/target batch by prefilter_threshold when client_filter is enabled.
        Returns (data_kept, targets_kept, kept_ratio).

        """
        self._prefilter__ensure_cfg()
        if not getattr(self, "client_filter", False):
            return data, targets, None


        probs = self._prefilter__score_tensor_batch(data.detach())
        keep = (probs < self.prefilter_threshold).numpy()  # True=keep

        B = data.size(0)
        kept = int(keep.sum())
        kept_ratio = kept / max(1, B)


        if kept == 0:
            keep[0] = True
            kept = 1
            kept_ratio = kept / max(1, B)

        keep_idx = torch.from_numpy(keep).to(data.device)
        data2 = data[keep_idx]
        targets2 = targets[keep_idx]
        return data2, targets2, kept_ratio

    def create_model(self):
        local_model=None
        target_model=None
        if self.params['type']==config.TYPE_CIFAR:
            local_model = ResNet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = ResNet18(name='Target',
                                   created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_MNIST:
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_TINYIMAGENET:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])

        local_model=local_model.to(device)
        target_model=target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0]) #for cifar: 5000
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        return per_participant_list

    def draw_dirichlet_plot(self,no_classes,no_participants,image_nums,alpha):
        fig= plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par=[0,10,20,30]
        for k in range(no_participants):
        # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            left += s[k]
        plt.legend(ncol=20,loc='lower left',  bbox_to_anchor=(0, 1),fontsize=4)
        plt.xlabel("Number of Images", fontsize=16)
        fig.tight_layout(pad=0.1)
        fig.savefig(self.folder_path+'/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds))

    def load_data(self):
        logger.info('Loading data')
        dataPath = './data'
        if self.params['type'] == config.TYPE_CIFAR:
            ### data load
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)

        elif self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                               ]))
            self.test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                ]))
        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            _data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                ]),
            }
            _data_dir = './data/tiny-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                    _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                   _data_transforms['val'])
            logger.info('reading data done')

        self.classes_dict = self.build_classes_dict()
        logger.info('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'], #100
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            ## sample indices for participants that are equally
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_data_poison ,self.test_targetlabel_data = self.poison_test_dataset()

        self.advasarial_namelist = self.params['adversary_list']

        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        self.benign_namelist =list(set(self.participants_list) - set(self.advasarial_namelist))



    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt,adversarial_index=-1, evaluation=False):

        images, targets = bptt

        poison_count= 0
        new_images=images.clone()
        new_targets=targets.clone()

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                poison_count+=1

            else: # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]


        try:
            self._prefilter__ensure_cfg()
            apply_all = (self.prefilter_apply_to == 'all')
            is_adv_round = (adversarial_index is not None and adversarial_index != -1)

            if (not evaluation) and self.client_filter and (apply_all or is_adv_round) and (not self.params.get('enable_firs_gate', False)):
                B = new_images.size(0)
                poison_k = min(self.params.get('poisoning_per_batch', 0), B)

                if poison_k > 0:
                    poisoned_imgs = new_images[:poison_k]
                    with torch.no_grad():
                        scores = self._prefilter__score_tensor_batch(poisoned_imgs)
                    keep_mask = (scores < self.prefilter_threshold)
                    drop_mask = ~keep_mask
                    drop_count = int(drop_mask.sum().item())

                    if drop_count > 0:

                        keep_poison_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).tolist()
                        keep_indices = keep_poison_idx + list(range(poison_k, B))
                        new_images = new_images[keep_indices]
                        new_targets = new_targets[keep_indices]
                        poison_count = len(keep_poison_idx)
                        print(f"[PREFILTER][IN-BATCH] drop {drop_count}/{poison_k} poisoned "
                              f"at client {adversarial_index}")
        except Exception as e:
            print(f"[PREFILTER][IN-BATCH] WARNING: failed with error: {e}")


        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

    def add_pixel_pattern(self, image, adversarial_index):
        """
        Apply a white-pixel trigger in tensor space.
        If Normalize(mean, std) is present, write (1 - mean) / std.
        Otherwise write 1.0 in the [0, 1] pixel domain.
        adversarial_index == -1 merges all trigger fragments.
        """
        img = image.clone()


        if adversarial_index == -1:
            coords = []
            for i in range(self.params.get('trigger_num', 1)):
                coords += self.params[f"{i}_poison_pattern"]
        else:
            coords = self.params[f"{adversarial_index}_poison_pattern"]


        def _find_norm(tfm):
            if tfm is None:
                return None
            if isinstance(tfm, transforms.Normalize):
                return (tfm.mean, tfm.std)
            if hasattr(tfm, "transforms"):  # Compose
                for s in tfm.transforms:
                    r = _find_norm(s)
                    if r is not None:
                        return r
            return None

        norm = _find_norm(getattr(getattr(self, "train_dataset", None), "transform", None))


        C, H, W = img.shape
        if norm is not None:
            mean, std = norm
            if not isinstance(mean, (list, tuple)): mean = [mean]*C
            if not isinstance(std,  (list, tuple)): std  = [std]*C
            for (x, y) in coords:
                if 0 <= x < H and 0 <= y < W:
                    for c in range(min(C, len(mean))):
                        img[c, x, y] = (1.0 - float(mean[c])) / float(std[c])
            return img
        else:
            for (x, y) in coords:
                if 0 <= x < H and 0 <= y < W:
                    img[:, x, y] = 1.0
            return img.clamp_(0.0, 1.0)

    def add_pixel_pattern(self, image, adversarial_index):
        """
        Apply a DBA trigger fragment. The default white_patch path preserves the
        original DBA behavior; other trigger families are opt-in by parameter.
        """
        coords = self._trigger_coords(adversarial_index)
        norm = self._find_normalize_stats(getattr(getattr(self, "train_dataset", None), "transform", None))
        mean, std = norm if norm is not None else (None, None)
        cfg = TriggerConfig(
            trigger_type=str(self.params.get("attack_eval_trigger_type", self.params.get("trigger_type", "white_patch"))).lower(),
            color=str(self.params.get("trigger_color", "white")).lower(),
            alpha=float(self.params.get("trigger_alpha", 1.0)),
            intensity=float(self.params.get("trigger_intensity", 1.0)),
            jitter=int(self.params.get("trigger_jitter", 0)),
            size_delta=int(self.params.get("trigger_size_delta", 0)),
            randomize=bool(self.params.get("trigger_randomize", False)),
        )
        return apply_trigger_fragment(image, coords, cfg, mean=mean, std=std)

    def _trigger_coords(self, adversarial_index):
        if adversarial_index == -1:
            coords = []
            for i in range(self.params.get('trigger_num', 1)):
                coords += self.params[f"{i}_poison_pattern"]
            return coords
        return self.params[f"{adversarial_index}_poison_pattern"]

    def _find_normalize_stats(self, tfm):
        if tfm is None:
            return None
        if isinstance(tfm, transforms.Normalize):
            return (tfm.mean, tfm.std)
        if hasattr(tfm, "transforms"):
            for step in tfm.transforms:
                result = self._find_normalize_stats(step)
                if result is not None:
                    return result
        return None


if __name__ == '__main__':
    np.random.seed(1)
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars= list(range(100))
    # show the data distribution among all participants.
    count_all= 0
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count=0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets= batch
            for t in targets:
                cifar_class_count[t.item()]+=1
            count += len(targets)
        count_all+=count
        print(par, cifar_class_count,count,max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all*1.0/100)
