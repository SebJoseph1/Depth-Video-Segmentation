import math
from dataclasses import dataclass
import re
import numpy as np
import os
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
    focal_length_real: str

    @property
    def sequence(self):
        return self.seq_id

    @property
    def id(self):
        return self.seq_id + "_" + self.frame_id

    @property
    def focal_length(self):
        return self.focal_length_real

    @staticmethod
    def from_filename(filename: str):
        match = re.match(r"^(\d+)_(\d+)_depth_(\d+\.\d+).png$", filename, re.I)
        return SemKittiSample(match.group(1), match.group(2), match.group(3))


@dataclass
class SemKittiClass:
    name: str
    ID: int
    hasInstances: bool
    color: Tuple[int, int, int]


classes = {
    #                 name                     ID       hasInstances    color
    0: SemKittiClass('car', 0, True, (0, 0, 255)),
    1: SemKittiClass('bicycle', 1, True, (245, 150, 100)),
    2: SemKittiClass('motorcycle', 2, True, (245, 230, 100)),
    3: SemKittiClass('truck', 3, True, (250, 80, 100)),
    4: SemKittiClass('other-vehicle', 4, True, (150, 60, 30)),
    5: SemKittiClass('person', 5, True, (111, 74, 0)),
    6: SemKittiClass('bicyclist', 6, True, (81, 0, 81)),
    7: SemKittiClass('motorcyclist', 7, True, (128, 64, 128)),
    8: SemKittiClass('road', 8, False, (244, 35, 232)),
    9: SemKittiClass('parking', 9, False, (250, 170, 160)),
    10: SemKittiClass('sidewalk', 10, False, (230, 150, 140)),
    11: SemKittiClass('other-ground', 11, False, (70, 70, 70)),
    12: SemKittiClass('building', 12, False, (102, 102, 156)),
    13: SemKittiClass('fence', 13, False, (190, 153, 153)),
    14: SemKittiClass('vegetation', 14, False, (180, 165, 180)),
    15: SemKittiClass('trunk', 15, False, (150, 100, 100)),
    16: SemKittiClass('terrain', 16, False, (150, 120, 90)),
    17: SemKittiClass('pole', 17, False, (153, 153, 153)),
    18: SemKittiClass('traffic-sign', 18, False, (50, 120, 255)),
    255: SemKittiClass('unlabeled', 255, False, (0, 0, 0)),
}


class SemKittiDataset(Dataset):

    def __init__(self, dir_input: str, classes: Dict, max_batch_size: int = 32, shuffle: bool = False):
        super().__init__()
        self.dir_input = dir_input
        self.classes = classes
        self.items = []
        self.sequences = {}
        self.max_batch_size = max_batch_size
        file_list = os.listdir(self.dir_input)
        file_list.sort()
        for filename in file_list:
            if "depth" in filename:
                current_frame = SemKittiSample.from_filename(filename)
                self.items.append(current_frame)
                if current_frame.sequence not in self.sequences:
                    self.sequences[current_frame.sequence] = []
                self.sequences[current_frame.sequence].append(len(self.items) - 1)
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
            file_name = f'{sample.id}_leftImg8bit.png'
        elif image_type == "d":
            file_name = f'{sample.id}_depth_{sample.focal_length}.png'
        elif image_type == "s":
            file_name = f'{sample.id}_gtFine_class.png'
        elif image_type == "in":
            file_name = f'{sample.id}_gtFine_instance.png'
        else:
            raise Exception("image_type incorrect")
        path = os.path.join(self.dir_input, file_name)
        return Image.open(path)#.resize((128, 128))

    def transform(self, img: Tuple[Image.Image], mask: List[Image.Image]):
        img_1 = TF.to_tensor(img[0])
        img_2 = TF.to_tensor(img[1])
        img = torch.cat([img_1, img_2], 0)
        truth = []
        for truth_image in mask:
            truth.append(TF.to_tensor(truth_image))
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
        img_segment = Image.fromarray(img_segment)

        return [img_depth, img_segment, img_instance]


if __name__ == "__main__":
    d = SemKittiDataset(
        dir_input="semkitti-dvps/video_sequence/train",
        classes=classes)
    print(d[0])
    print(len(d))
    exit()
    for i in range(50):
        g = random.choice(d)
        k = d.to_image(g[1])
        # for t in k:
        #     t.show()