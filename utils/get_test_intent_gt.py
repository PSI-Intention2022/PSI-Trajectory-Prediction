from opts import get_opts
from datetime import datetime
import os
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from train import train_intent
from test import validate_intent, test_intent
from utils.log import RecordResults


def main(args):
    writer = SummaryWriter(args.checkpoint_path)
    recorder = RecordResults(args)
    ''' 1. Load database '''
    if not os.path.exists(os.path.join(args.database_path, 'intent_database_train.pkl')):
        create_database(args)
    else:
        print("Database exists!")
    train_loader, val_loader, test_loader = get_dataloader(args)
    get_intent_gt(test_loader, '../test_gt/val_intent_gt.json', args)
    get_intent_gt(test_loader, '../test_gt/test_intent_gt.json', args)

def get_intent_gt(dataloader, output_path, args):
    dt = {}
    for itern, data in enumerate(dataloader):
        # if args.intent_type == 'mean' and args.intent_num == 2:  # BCEWithLogitsLoss
        #     gt_intent = data['intention_binary'][:, args.observe_length]
        #     gt_intent_prob = data['intention_prob'][:, args.observe_length]
        # print(data.keys())
        # print(data['frames'])
        for i in range(len(data['frames'])):
            vid = data['video_id'][i] # str list, bs x 60
            pid = data['ped_id'][i] # str list, bs x 60
            fid = (data['frames'][i][-1]+1).item() # int list, bs x 15, observe 0~14, predict 15th intent
            gt_int = data['intention_binary'][i][args.observe_length].item() # int list, bs x 60
            gt_int_prob = data['intention_prob'][i][args.observe_length].item()  # float list, bs x 60
            gt_disgr = data['disagree_score'][i][args.observe_length].item() # float list, bs x 60

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['intent'] = gt_int
            # dt[vid][pid][fid]['intent_prob'] = gt_int_prob
            # dt[vid][pid][fid]['disagreement'] = gt_disgr

    with open(output_path, 'w') as f:
        json.dump(dt, f)

def get_intent_reasoning_gt():
    pass



if __name__ == '__main__':
    args = get_opts()
    # Task
    args.datset = 'PSI200'
    args.task_name = 'ped_intent'
    args.model_name = 'lstm_int_bbox' # LSTM module, with bboxes sequence as input, to predict intent

    # Model
    args.load_image = False # only bbox input
    if args.load_image:
        args.backbone = 'resnet'
        args.freeze_backbone = False
    else:
        args.backbone = None
        args.freeze_backbone = False
    args.loss_weights = {
        'loss_intent': 1.0,
        'loss_traj': 1.0,
        'loss_driving': 1.0
    }
    # Data - intent prediction
    args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
    args.intent_type = 'mean'
    args.intent_loss = ['bce']
    args.intent_disagreement = 1 # -1: not use disagreement 1: use disagreement to reweigh samples
    args.intent_positive_weight = 0.5 # reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)

    # Data - trajectory
    args.traj_loss = ['bbox_l1']
    args.normalize_bbox = None
    # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
    # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]

    # Train
    args.epochs = 10
    args.batch_size = 128
    args.lr = 1e-3
    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    # Record
    now = datetime.now()
    time_folder = now.strftime('%Y%m%d%H%M%S')
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name, args.dataset, args.model_name, time_folder)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    with open(os.path.join(args.checkpoint_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    main(args)