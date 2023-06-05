import collections

from test import validate_traj
import torch
import numpy as np
import os

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def train_traj(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer):
    pos_weight = torch.tensor(args.intent_positive_weight).to(device) # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    criterions = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight).to(device),
        'MSELoss': torch.nn.MSELoss(reduction='none').to(device),
        'BCELoss': torch.nn.BCELoss().to(device),
        'CELoss': torch.nn.CrossEntropyLoss().to(device),
        'L1Loss': torch.nn.L1Loss().to(device),
    }
    epoch_loss = {'loss_intent': [], 'loss_traj': []}

    for epoch in range(args.epochs):
        niters = len(train_loader)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_traj_epoch(epoch, model, optimizer, criterions, epoch_loss, train_loader, args, recorder, writer)
        scheduler.step()

        if epoch % 1 == 0:
            print(f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                  f"loss_intent = {np.mean(epoch_loss['loss_intent']): .4f}, "
                  f"loss_traj = {np.mean(epoch_loss['loss_traj']): .4f}")

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = len(val_loader)
            recorder.eval_epoch_reset(epoch, niters)
            validate_traj(epoch, model, val_loader, args, recorder, writer)

        torch.save(model.state_dict(), args.checkpoint_path + f'/latest.pth')


def train_traj_epoch(epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        optimizer.zero_grad()
        traj_pred = model(data)
        # intent_pred: sigmoid output, (0, 1), bs
        # traj_pred: logit, bs x ts x 4

        traj_gt = data['bboxes'][:, args.observe_length:, :].type(FloatTensor)
        bs, ts, _ = traj_gt.shape
        # center: bs x ts x 2
        # traj_center_gt = torch.cat((((traj_gt[:, :, 0] + traj_gt[:, :, 2]) / 2).unsqueeze(-1),
        #                             ((traj_gt[:, :, 1] + traj_gt[:, :, 3]) / 2).unsqueeze(-1)), dim=-1)
        # traj_center_pred = torch.cat((((traj_pred[:, :, 0] + traj_pred[:, :, 2]) / 2).unsqueeze(-1),
        #                               ((traj_pred[:, :, 1] + traj_pred[:, :, 3]) / 2).unsqueeze(-1)), dim=-1)

        loss_traj = torch.tensor(0.).type(FloatTensor)
        if 'bbox_l1' in args.traj_loss:
            loss_bbox_l1 = torch.mean(criterions['L1Loss'](traj_pred, traj_gt))
            batch_losses['loss_bbox_l1'].append(loss_bbox_l1.item())
            loss_traj += loss_bbox_l1

        loss = args.loss_weights['loss_traj'] * loss_traj
        loss.backward()
        optimizer.step()

        # Record results
        batch_losses['loss'].append(loss.item())
        batch_losses['loss_traj'].append(loss_traj.item())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                  f"loss_traj = {np.mean(batch_losses['loss_traj']): .4f}, ")
        recorder.train_traj_batch_update(itern, data, traj_gt.detach().cpu().numpy(), traj_pred.detach().cpu().numpy(),
                                         loss.item(), loss_traj.item())

    epoch_loss['loss_traj'].append(np.mean(batch_losses['loss_traj']))

    recorder.train_traj_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f'LearningRate', optimizer.param_groups[-1]['lr'], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f'Losses/{key}', np.mean(val), epoch)

    return epoch_loss