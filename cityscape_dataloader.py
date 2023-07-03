import math
from dataclasses import dataclass
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Tuple, List
import random
import torchvision.transforms.functional as TF


@dataclass
class SemKittiSample:
    seq_id: str
    frame_id: str
    city: str
    focal_length_real1: str
    focal_length_real2: str

    @property
    def id(self):
        return self.seq_id + "_" + self.frame_id
    
    @property
    def focal_length(self):
        return self.focal_length_real1 + "_" +self.focal_length_real2

    @staticmethod
    def from_filename(filename: str):
        match = re.match(r"^(\d+)_(\d+)_(\w+)_(\d+)_(\d+)_depth.png$", filename, re.I)
        return SemKittiSample(match.group(1), match.group(2),match.group(3),match.group(4),match.group(5))


@dataclass
class SemKittiClass:
    name: str
    ID: int
    hasInstances: bool
    color: Tuple[int, int, int]


classes = {
    #                 name                     ID       hasInstances    color
    0: SemKittiClass('road', 0, False, (128,64,128)),
    1: SemKittiClass('sidewalk', 1, False, (244,35,232)),
    2: SemKittiClass('building', 2, False, (70,70,70)),
    3: SemKittiClass('wall', 3, False, (102,102,156)),
    4: SemKittiClass('fence', 4, False, (190,153,153)),
    5: SemKittiClass('pole', 5, False, (153,153,153)),
    6: SemKittiClass('traffic light', 6, False, (250,170,30)),
    7: SemKittiClass('traffic sign', 7, False, (220,220,0)),
    8: SemKittiClass('vegetation', 8, False, (107,142,35)),
    9: SemKittiClass('terrain', 9, False, (152,251,152)),
    10: SemKittiClass('sky', 10, False, (70,130,180)),
    11: SemKittiClass('person', 11, True, (220,20,60)),
    12: SemKittiClass('rider', 12, True, (255,0,0)),
    13: SemKittiClass('car', 13, True, (0,0,142)),
    14: SemKittiClass('truck', 14, True, (0,0,70)),
    15: SemKittiClass('bus', 15, True, (0,60,100)),
    16: SemKittiClass('train', 16, True, (0,80,100)),
    17: SemKittiClass('motorcycle', 17, True, (0,0,230)),
    18: SemKittiClass('bicycle', 18, True, (119,11,32)),
    32: SemKittiClass('unlabeled', 32, False, (255, 255, 255)),
}


class SemKittiDataset(Dataset):

    def __init__(self, dir_input: str, classes: Dict, max_batch_size: int = 32, shuffle: bool = False):
        super().__init__()
        self.dir_input = dir_input
        self.classes = classes
        self.items = []
        self.sequences = {}
        self.sequeeze_id = {}
        self.current_id = 1
        self.max_batch_size = max_batch_size

        file_list = os.listdir(self.dir_input)
        file_list.sort()
        for filename in file_list:
            if "depth" in filename:
                current_frame = SemKittiSample.from_filename(filename)
                self.items.append(current_frame)
                if current_frame.seq_id not in self.sequences:
                    self.sequences[current_frame.seq_id] = []
                self.sequences[current_frame.seq_id].append(len(self.items) - 1)
        assert len(self.items) > 0, f"No items found in {self.dir_input}"
        # This gets precomputed to save computation for later
        self.batches = []
        for s in self.sequences:
            self.batches.append([])
            for item_index in self.sequences[s]:
                if len(self.batches[-1]) >= self.max_batch_size + 1:
                    self.batches.append([])
                self.batches[-1].append(item_index)
        self.shuffle_indexes = list(range(len(self.batches)))
        if shuffle:
            random.shuffle(self.shuffle_indexes)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i: int):
        batch_index = self.shuffle_indexes[i]
        if len(self.batches[batch_index]) < 2:
            if i < len(self.batches):
                batch_index = self.shuffle_indexes[i+1]
            else:
                batch_index = self.shuffle_indexes[i-1]
        i = batch_index
        self.sequeeze_id = {}
        self.current_id = 1
        sequence = [[], [[], [], []]]
        for index, item in enumerate(self.batches[i]):
            if index == len(self.batches[i]) - 1:
                continue
            sample_1 = self.items[item]
            sample = self.items[item + 1]
            input_1 = self.load_file(sample_1, "i")
            input = self.load_file(sample, "i")
            truth_depth = self.load_file(sample, "d")
            truth_segment = self.load_file(sample, "s")
            truth_instance = self.load_file(sample, "in")
            try:
                element = self.transform((input, input_1), [truth_depth, truth_segment, truth_instance])
            except:
                print(sample, sample_1, item)
                exit()
            sequence[0].append(element[0])
            sequence[1][0].append(element[1][0])
            sequence[1][1].append(element[1][1])
            sequence[1][2].append(element[1][2])
        to_ret = [torch.stack(sequence[0]), [torch.stack(sequence[1][0]), torch.stack(sequence[1][1]), torch.stack(sequence[1][2])]]
        print('Sequence: ' + str(self.items[self.batches[i][0]].seq_id))
        return to_ret

    def load_file(self, sample: SemKittiSample, image_type: str) -> Image:
        file_name = ""
        if image_type == "i":
            file_name = f'{sample.id}_{sample.city}_{sample.focal_length}_leftImg8bit.png'
        elif image_type == "d":
            file_name = f'{sample.id}_{sample.city}_{sample.focal_length}_depth.png'
        elif image_type == "s":
            file_name = f'{sample.id}_{sample.city}_{sample.focal_length}_gtFine_instanceTrainIds.png'
        elif image_type == "in":
            file_name = f'{sample.id}_{sample.city}_{sample.focal_length}_gtFine_instanceTrainIds.png'
        else:
            raise Exception("image_type incorrect")
        path = os.path.join(self.dir_input, file_name)
        return Image.open(path)#.resize((128, 128))

    def transform(self, img: Tuple[Image.Image], mask: List[Image.Image]):
        img_1 = TF.to_tensor(img[0])
        img_2 = TF.to_tensor(img[1])
        img = torch.cat([img_1, img_2], 0)
        truth = []
        truth.append(TF.to_tensor(mask[0]))
        panoptic_mask = TF.to_tensor(mask[1])
        instance_mask = torch.zeros_like(panoptic_mask)
        semantic_mask = panoptic_mask / 1000
        semantic_mask = semantic_mask.to(torch.int32).to(torch.float32)
        truth.append(semantic_mask)
        for panoptic_id in torch.unique(panoptic_mask):
            if panoptic_id.item() not in self.sequeeze_id.keys():
                self.sequeeze_id[panoptic_id.item()] = self.current_id
                self.current_id += 1
            if int(panoptic_id/ 1000) in thing_list:
                instance_mask[panoptic_mask == panoptic_id] = self.sequeeze_id[panoptic_id.item()]
        truth.append(instance_mask)
        return img, truth

    def to_image(self, indices: List[torch.Tensor]) -> List[Image.Image]:
        img_depth = TF.to_pil_image(indices[0].cpu(), 'I')
        img_instance = np.zeros([indices[2]
                                .shape[1], indices[2]
                                .shape[2], 3], dtype=np.uint8)
        colors = {0: np.array([0, 0, 0])}
        for i in range(indices[2].shape[1]):
            for j in range(indices[2].shape[2]):
                instance = indices[2][0][i][j].item()
                # print(instance)
                if instance not in colors:
                    colors[instance] = np.array(
                        [random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)])
                img_instance[i][j] = colors[instance]
        img_instance = Image.fromarray(img_instance)

        img_segment = np.zeros([indices[1].shape[1], indices[1].shape[2], 3], dtype=np.uint8)
        for i in range(indices[1].shape[1]):
            for j in range(indices[1].shape[2]):
                class_id = indices[1][0][i][j].item()
                img_segment[i][j] = classes[class_id].color
        # img_segment = Image.fromarray(img_segment)

        return [img_depth, img_segment, img_instance]

thing_list = list(range(11,19))
if __name__ == "__main__":
    d = SemKittiDataset(
        dir_input="cityscapes-dvps/video_sequence/train",
        classes=classes,max_batch_size=32)
    
    print(len(d))
    
    for i in range(1):
        # print(d[i],i)
        for j in range(len(d[i][0])):
            # cv2.imshow("Image",cv2.cvtColor(d[i][0][j][:3,:,:].permute(1,2,0).cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
            g = d.to_image([d[i][1][0][j],d[i][1][1][j],d[i][1][2][j]])
            fig1, ax1 = plt.subplots(1, 2, figsize=(18, 6))
            ax1[0].title.set_text('Input Image')
            print(j," : ",torch.unique(d[i][1][1][j]))
            ax1[0].imshow(d[i][0][j][:3,:,:].permute(1,2,0).cpu().detach().numpy())
            ax1[1].title.set_text('Depth')
            ax1[1].imshow(g[1])
            # cv2.imshow("Segement",g[1][j])
        # g = random.choice(d)
        # k = d.to_image(g[1])
        # for t in k:
        #     t.show()