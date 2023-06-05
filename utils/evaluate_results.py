import json
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np

def evaluate_intent(groundtruth='', prediction='', args=None):
    with open(groundtruth, 'r') as f:
        gt_intent = json.load(f)

    with open(prediction, 'r') as f:
        pred_intent = json.load(f)

    gt = []
    pred = []
    for vid in gt_intent.keys():
        for pid in gt_intent[vid].keys():
            for fid in gt_intent[vid][pid].keys():
                gt.append(gt_intent[vid][pid][fid]['intent'])
                pred.append(pred_intent[vid][pid][fid]['intent_pred'])
    gt = np.array(gt)
    pred = np.array(pred)
    res = measure_intent_prediction(gt, pred, args)
    print('Acc: ', res['Acc'])
    print('F1: ', res['F1'])
    print('mAcc: ', res['mAcc'])
    print('ConfusionMatrix: ', res['ConfusionMatrix'])
    return res['F1']



def measure_intent_prediction(target, prediction, args):
    '''
    Here we only predict one 'intention' for one track (15 frame observation). (not a sequence as before)
    :param target: (bs x 1), hard label; target_prob: soft probability, 0-1, agreement mean([0, 0.5, 1]).
    :param prediction: (bs), sigmoid probability, 1-dim, should use 0.5 as threshold
    :return:
    '''
    print("Evaluating Intent ...")
    results = {
        'MSE': 0,
        'Acc': 0,
        'F1': 0,
        'mAcc': 0,
        'ConfusionMatrix': [[]],
    }

    bs = target.shape[0]
    # lbl_target = np.argmax(target, axis=-1) # bs x ts
    lbl_target = target # bs
    # lbl_taeget_prob = target_prob
    lbl_pred = np.round(prediction) # bs, use 0.5 as threshold

    # MSE = np.mean(np.square(lbl_taeget_prob - prediction))
    # hard label evaluation - acc, f1
    Acc = accuracy_score(lbl_target, lbl_pred) # calculate acc for all samples
    F1 = f1_score(lbl_target, lbl_pred, average='macro')

    intent_matrix = confusion_matrix(lbl_target, lbl_pred)  # [2 x 2]
    intent_cls_acc = np.array(intent_matrix.diagonal() / intent_matrix.sum(axis=-1)) # 2
    intent_cls_mean_acc = intent_cls_acc.mean(axis=0)

    # results['MSE'] = MSE
    results['Acc'] = Acc
    results['F1'] = F1
    results['mAcc'] = intent_cls_mean_acc
    results['ConfusionMatrix'] = intent_matrix
    return results

def evaluate_traj(groundtruth='', prediction='', args=None):
    with open(groundtruth, 'r') as f:
        gt_traj = json.load(f)

    with open(prediction, 'r') as f:
        pred_traj = json.load(f)

    gt = []
    pred = []
    for vid in gt_traj.keys():
        for pid in gt_traj[vid].keys():
            for fid in gt_traj[vid][pid].keys():
                gt.append(gt_traj[vid][pid][fid]['traj'])
                pred.append(pred_traj[vid][pid][fid]['traj'])
    gt = np.array(gt)
    pred = np.array(pred)
    traj_results = measure_traj_prediction(gt, pred, args)

    for key in ['ADE', 'FDE', 'ARB', 'FRB']: #, 'Bbox_MSE', 'Bbox_FMSE', 'Center_MSE', 'Center_FMSE']:
        for time in ['0.5', '1.0', '1.5']:
            val = traj_results[key][time]
            print(f'Eval/Results/{key}_{time}', val)

    score = np.mean([traj_results['ADE'][t] for t in ['0.5', '1.0', '1.5']])
    return score

def measure_traj_prediction(target, prediction, args):
    print("Evaluating Trajectory ...")
    target = np.array(target)
    prediction = np.array(prediction)
    assert target.shape[1] == args.predict_length
    assert target.shape[2] == 4  # bbox
    assert prediction.shape[1] == args.predict_length
    assert prediction.shape[2] == 4
    results = {
        # 'Bbox_MSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        # 'Bbox_FMSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        # 'Center_MSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        # 'Center_FMSE': {'0.5': 0, '1.0': 0, '1.5': 0},
        'ADE': {'0.5': 0, '1.0': 0, '1.5': 0},  # center
        'FDE': {'0.5': 0, '1.0': 0, '1.5': 0},  # center
        'ARB': {'0.5': 0, '1.0': 0, '1.5': 0},  # bbox - B: bbox
        'FRB': {'0.5': 0, '1.0': 0, '1.5': 0},  # bbox - B: bbox
    }
    bs, ts, _ = target.shape
    performance_MSE = np.square(target - prediction).sum(axis=2)  # n_samples x ts x 4 --> bs x ts
    performance_RMSE = np.sqrt(performance_MSE)  # bs x ts
    for t in [0.5, 1.0, 1.5]:
        end_frame = int(t * args.fps)
    #     # 1. bbox MSE
    #     results['Bbox_MSE'][str(t)] = performance_MSE[:, :end_frame].mean(axis=None)
    #     # 2. bbox FMSE
    #     results['Bbox_FMSE'][str(t)] = performance_MSE[:, end_frame - 1].mean(axis=None)
    #
        # 5. ARB - bbox
        results['ARB'][str(t)] = performance_RMSE[:, :end_frame].mean(axis=None)
        # 6. FRB - bbox
        results['FRB'][str(t)] = performance_RMSE[:, end_frame - 1].mean(axis=None)

    # centers
    center_target = np.zeros((bs, ts, 2))
    center_pred = np.zeros((bs, ts, 2))
    for i in range(bs):
        for j in range(ts):
            center_target[i, j, 0] = (target[i, j, 0] + target[i, j, 2]) / 2
            center_target[i, j, 1] = (target[i, j, 1] + target[i, j, 3]) / 2
            center_pred[i, j, 0] = (prediction[i, j, 0] + prediction[i, j, 2]) / 2
            center_pred[i, j, 1] = (prediction[i, j, 1] + prediction[i, j, 3]) / 2

    performance_CMSE = np.square(center_target - center_pred).sum(axis=2)  # bs x ts x 4 --> bs x ts
    performance_CRMSE = np.sqrt(performance_CMSE)  # bs x ts

    for t in [0.5, 1.0, 1.5]:
        end_frame = int(t * args.fps)
        # # 3. C_MSE
        # results['Center_MSE'][str(t)] = performance_CMSE[:, :end_frame].mean(axis=None)
        # # 4. C_FMSE
        # results['Center_FMSE'][str(t)] = performance_CMSE[:, end_frame - 1].mean(axis=None)
        # 7. ADE - center
        results['ADE'][str(t)] = performance_CRMSE[:, : end_frame].mean(axis=None)
        # 8. FDE - center
        results['FDE'][str(t)] = performance_CRMSE[:, end_frame - 1].mean(axis=None)

    return results




if __name__ == '__main__':
    args = None
    evaluate_intent('gt.json', 'pred.json', args)