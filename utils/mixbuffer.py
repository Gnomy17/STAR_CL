# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.augmentations import apply_transform
from utils.conf import get_device


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.

    Args:
        num_seen_examples: the number of seen examples
        buffer_size: the maximum buffer size

    Returns:
        the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

import numpy as np
class MixBuffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, mix_alpha=0.5, num_classes=10,device="cpu"):
        """
        Initialize a reservoir-based Buffer object.

        Args:
            buffer_size (int): The maximum size of the buffer.
            device (str, optional): The device to store the buffer on. Defaults to "cpu".

        Note:
            If during the `get_data` the transform is PIL, data will be moved to cpu and then back to the device. This is why the device is set to cpu by default.
        """
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        self.counter = [0] * buffer_size
        self.alpha = mix_alpha
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits']
        self.attention_maps = [None] * buffer_size

    def to(self, device):
        """
        Move the buffer and its attributes to the specified device.

        Args:
            device: The device to move the buffer and its attributes to.

        Returns:
            The buffer instance with the updated device and attributes.
        """
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        """
        Returns the number items in the buffer.
        """
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if (attr_str.endswith('els')) else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
        if hasattr(self, 'labels'):
            self.labels = torch.zeros_like(self.labels).float()
        

    @property
    def used_attributes(self):
        """
        Returns a list of attributes that are currently being used by the object.
        """
        return [attr_str for attr_str in self.attributes if hasattr(self, attr_str)]

    def add_data(self, examples, labels=None, logits=None, task_labels=None, attention_maps=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels

        Note:
            Only the examples are required. The other tensors are initialized only if they are provided.
        """
        if (labels is not None) and (labels.squeeze().dim() <= 1):
            labels = F.one_hot(labels, self.num_classes).float()
        elif (labels is not None):
            labels = labels.float() * 1.0
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits)
        
        ### step 1, just naively interpolate instead of replacing
        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            
            self.num_seen_examples += 1
            if index >= 0:
                lam = np.random.beta(self.alpha, self.alpha)
                if self.counter[index] == 0 or self.counter[index] >= 2:
                        self.examples[index] = examples[i]
                else:
                    self.examples[index] = lam * self.examples[index] + (1 - lam) * examples[i].to(self.device)
                if labels is not None:
                    if self.counter[index] == 0 or self.counter[index] >= 2:
                        self.labels[index] = labels[i]
                    else:
                        self.labels[index] = lam* self.labels[index] + (1-lam) * labels[i].to(self.device)
                if logits is not None:
                    if self.counter[index] == 0 or self.counter[index] >= 2:
                        self.logits[index] = logits[i]
                    else:
                        self.logits[index] = lam * self.logits[index] + (1-lam) * logits[i].to(self.device)
                if self.counter[index] >= 2:
                    self.counter[index] = 0
                self.counter[index] += 1
                # print(self.counter[index])

    def get_data(self, size: int, transform: nn.Module = None, return_index=False, device=None, mask_task_out=None, cpt=None) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            return_index: if True, returns the indexes of the sampled items
            mask_task: if not None, masks OUT the examples from the given task
            cpt: the number of classes per task (required if mask_task is not None and task_labels are not present)

        Returns:
            a tuple containing the requested items. If return_index is True, the tuple contains the indexes as first element.
        """
        target_device = self.device if device is None else device

        if mask_task_out is not None:
            assert hasattr(self, 'task_labels') or cpt is not None
            assert hasattr(self, 'task_labels') or hasattr(self, 'labels')
            samples_mask = (self.task_labels != mask_task_out) if hasattr(self, 'task_labels') else self.labels // cpt != mask_task_out

        num_avail_samples = self.examples.shape[0] if mask_task_out is None else samples_mask.sum().item()
        num_avail_samples = min(self.num_seen_examples, num_avail_samples)

        if size > min(num_avail_samples, self.examples.shape[0]):
            size = min(num_avail_samples, self.examples.shape[0])

        choice = np.random.choice(num_avail_samples, size=size, replace=False)
        if transform is None:
            def transform(x): return x

        selected_samples = self.examples[choice] if mask_task_out is None else self.examples[samples_mask][choice]
        ret_tuple = (apply_transform(selected_samples, transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                selected_attr = attr[choice] if mask_task_out is None else attr[samples_mask][choice]
                ret_tuple += (selected_attr.to(target_device),)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None, device=None) -> Tuple:
        """
        Returns the data by the given index.

        Args:
            index: the index of the item
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple containing the requested items. The returned items depend on the attributes stored in the buffer from previous calls to `add_data`.
        """
        target_device = self.device if device is None else device

        if transform is None:
            def transform(x): return x
        ret_tuple = (apply_transform(self.examples[indexes], transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None, device=None) -> Tuple:
        """
        Return all the items in the memory buffer.

        Args:
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple with all the items in the memory buffer
        """
        target_device = self.device if device is None else device
        if transform is None:
            def transform(x): return x

        ret_tuple = (apply_transform(self.examples[:len(self)], transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)[:len(self)].to(target_device)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

