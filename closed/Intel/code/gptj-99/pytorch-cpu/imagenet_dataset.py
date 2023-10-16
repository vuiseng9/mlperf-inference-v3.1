import torch
import numpy as np
import logging
import json
import copy
import os
import cv2
from PIL import Image
from vision_tools.dataset_utils import create_imagenet_valset_loader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DATASET")

#TODO: Remove this before submissoin
USE_RANDOM=False
import random
if USE_RANDOM:
    random.seed(9973)

MAX_SAMPLES=50000

class OutputItem:
    def __init__(self, query_id_list, result, array_type_code='B'):
        self.query_id_list = query_id_list
        self.result = result
        self.array_type_code = array_type_code
        self.receipt_time = None
        self.outqueue_time = None

    def set_receipt_time(self, receipt_time):
        self.receipt_time = receipt_time

    def set_outqueued_time(self, outqueue_time):
        self.outqueue_time = outqueue_time


class Dataset(object):
    """ Dataset class for gpt-j """

    def __init__(self, dataset_path=None, total_sample_count=MAX_SAMPLES):
        self.dataset_path = dataset_path
        self.ground_truth_file = os.path.join(self.dataset_path, "val_map.txt")
        # if not os.path.exists(self.ground_truth_file):
        #     raise FileNotFoundError(self.ground_truth_file)
        self.total_sample_count = total_sample_count
        self.image_list = []
        # self.label_list = []
        self.dataset = []
        self.targets = []

    def loadDataset(self):
        """ Loads the dataset into memory """

        # with open(self.ground_truth_file, "r") as fid:
        #     for line in fid:
        #         imagefile, label = line.strip().split()
        #         self.image_list.append(imagefile)
        #         self.targets.append(int(label))

        # self.total_sample_count = min(self.total_sample_count, len(self.image_list))
        
        val_loader = create_imagenet_valset_loader(self.dataset_path, 1)

        for i, (input_, target) in enumerate(val_loader):
            # np_inputs, np_targets = input_.numpy(), target.numpy()
            # print(i, target)
            self.dataset.append(input_)
            self.targets.append(target)
            if i+1 == self.total_sample_count:
                break

    def getInputLengths(self):
        # dummy value
        return []
    
    def getWarmupSamples(self):
        """ Loads samples to use for warmup """
        return torch.rand(1, 3, 224, 224)
    
    def postProcess(self, query_id_list, sample_index_list, results):
        """ Postprocesses the predicted output
             output:    .output_tokens tensor
                        .input_seq_lens
        """
        processed_results = []
        results = torch.argmax(results, axis=1)
        n = results.shape[0]
        for idx in range(n):
            result = results[idx]
            processed_results.append(result.numpy())
        # print(processed_results)
        return OutputItem(query_id_list, processed_results, array_type_code='B')


    def __getitem__(self, index):
        """ Returns sample at 'index' """
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def getSamples(self, sample_index_list): # Assumes batch size 1
        """ Returns samples given 'sample_index_list' """
        if len(sample_index_list)==1:
            return self.dataset[sample_index_list[0]]
            
        batch_img_list = []
        for index in sample_index_list:
            batch_img_list.append(self.dataset[index])
        return torch.vstack(batch_img_list)