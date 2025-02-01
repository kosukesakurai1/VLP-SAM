r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
# from data.coco2pascal import DatasetCOCO2PASCAL


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, model):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            # 'coco2pascal': DatasetCOCO2PASCAL
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])
        
        if model.module.backbone_type in ['ViT-L/14', 'RN50', 'ViT-B/16', 'CS-ViT-B/16', 'CS-ViT-L/14', 'CS-RN50']:
            cls.vlmtransform = model.module.clippreprocess
        else:
            cls.vlmtransform = cls.transform

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, model, args, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize, vlmtransform=cls.vlmtransform, model=model, args=args)
        if split == 'trn':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=shuffle)
            shuffle = False
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=shuffle)
            pin_memory = True
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, pin_memory=True, num_workers=nworker, sampler=sampler)

        return dataloader