import numpy as np
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import clip
# from pytorch_loss import FocalLossV1
from datetime import datetime
from sklearn.metrics import classification_report

from copy import deepcopy
from session_settings import shapenet2modelnet, shapenet2co3d
from random import random
from datasets.CILdataset import *
from torch.nn.parallel import DataParallel

from models import FewShotCIL, focal_loss, FewShotCILwoRn2, FewShotCILwPoint

import sys, importlib
sys.path.append('models')

#################################################### TConfigurations ############################################
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device_id = [0, 1]
exp_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
"""
Arguments
"""
parser = argparse.ArgumentParser()
# add argss
args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
"""main"""


""" ↓↓↓↓↓session setup↓↓↓↓↓ """
session_maker = shapenet2co3d()
id2name = session_maker.get_id2name()
""" ↑↑↑↑↑session setup↑↑↑↑↑ """


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

def train_loop(dataloader, model):
    model.train()
    for points, label in tqdm(dataloader):
        label = label.to(device)
        points = points.to(device)
        
        # Compute prediction
        # Compute loss
        # update metrics
        # Backpropagation
  
def test_loop(dataloader, model):
    with torch.no_grad():
        for points, label in tqdm(dataloader):
            label = label.to(device)
            points = points.to(device)
            
            # Compute prediction
            
    # calculate score of session_i


def train():
    dataset_train_0, dataset_test_0 = session_maker.make_session(session_id=0, update_memory=args.memory_shot)
    num_cat_0 = dataset_test_0.get_cat_num()
    train_loader_0 = torch.utils.data.DataLoader(dataset_train_0, batch_size=args.batch_size, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, shuffle=True, persistent_workers=True, drop_last=True)
    test_loader_0 = torch.utils.data.DataLoader(dataset_test_0, batch_size=args.batch_size, num_workers=args.workers,
                                                    pin_memory=args.pin_memory, shuffle=True, persistent_workers=True, drop_last=True)  
    
    # init your model
    model = None
    
    """ ================Main Training============== """
    for t in range(args.epoch0):
        train_loop(train_loader_0, model)
    test_loop(test_loader_0, model)
    
    # incremental tasks`    `
    for task_id in range(1, session_maker.tot_session()):
        dataset_train_i, dataset_test_i = session_maker.make_session(session_id=task_id, update_memory=args.memory_shot)
        num_cat_i = dataset_test_i.get_cat_num()
        train_loader_i = torch.utils.data.DataLoader(dataset_train_i, batch_size=args.batch_size, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, shuffle=True, persistent_workers=True, drop_last=True)
        test_loader_i = torch.utils.data.DataLoader(dataset_test_i, batch_size=args.batch_size, num_workers=args.workers,
                                                        pin_memory=args.pin_memory, shuffle=True, persistent_workers=True, drop_last=True)
        for t in range(args.epochi):
            train_loop(train_loader_i, model)
        test_loop(test_loader_i, model)
        

if __name__ == "__main__":
    train()

