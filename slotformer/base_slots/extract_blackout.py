import os
import sys
import pdb
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data._utils.collate import default_collate

from nerv.utils import dump_obj, mkdir_or_exist

from models import build_model
from datasets import build_dataset, build_clevrer_dataset, build_adept_dataset
from nerv.training import BaseDataModule
from vp_utils import pred_eval_step, postproc_mask, masks_to_boxes, PALETTE_torch
import lpips
import pickle

@torch.no_grad()

def get_input(params, data_dict):
    """Prepare burn-in frames/gt data.""" 
    history_len = 6
    rollout_len = 42
    gt = data_dict['img'][:, history_len:]

    if 'mask' in data_dict:
        gt_mask = data_dict['mask'][:, history_len:].long()
    else:
        gt_mask = None
    if 'bbox' in data_dict:
        gt_bbox = data_dict['bbox'][:, history_len:]
        gt_pres_mask = data_dict['pres_mask'][:, history_len:].bool()
    else:
        gt_bbox, gt_pres_mask = None, None

    assert gt.shape[1] == rollout_len

    return gt, gt_mask, gt_bbox, gt_pres_mask

def get_output(params, out_dict):
    """Extract outputs for evaluation."""
    history_len = 6
    rollout_len = 42

    if params.model == 'StoSAVi':
        pred = out_dict['post_recon_combined'][:, history_len:]
        pred_mask = postproc_mask(out_dict['post_masks'])[:, history_len:]
        pred_bbox = masks_to_boxes(pred_mask, params.slot_dict['num_slots'])
    else:
        raise NotImplementedError(f'Unknown model: {params.model}')

    assert pred.shape[1] == rollout_len
    if pred_mask is not None:
        assert pred_mask.shape[1] == rollout_len
    if pred_bbox is not None:
        assert pred_bbox.shape[1] == rollout_len

    return pred, pred_mask, pred_bbox

def append_statistics(memory1, memory2, ignore=[]):
    for key in memory1:
        if key not in ignore:
            memory2[key].append(memory1[key])
    return memory2

def _save_video(videos, video_fn, dim=3):
    """Save torch tensors to a video."""
    video = torch.cat(videos, dim=dim)  # [T, 3, 2*H, B*W]
    video = (video * 255.).numpy().astype(np.uint8)
    save_path = os.path.join('vis',
                             params.dataset.split('_')[0], args.params,
                             video_fn)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_video(video, save_path, fps=4)


def test_blackout(model):
    model.eval()
    torch.cuda.empty_cache()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    # val_loading
    val_set = build_dataset(params, val_only=True)
    val_set.load_mask = True
    datamodule = BaseDataModule(params, train_set=val_set, val_set=val_set, use_ddp=False)
    val_loader = datamodule.val_loader
    blackout_p = 0.2
    evaluation_mode = 'blackout'

    metric_complete = {'mse': [], 'ssim': [], 'psnr': [], 'percept_dist': [], 'ari': [], 'fari': [], 'miou': [], 'ap': [], 'ar': [], 'blackout': []}

    # videos
    video_dic = {'gt' : [], 'pred' : [], 'input': [], 'blackout': []}
    num_videos = 4
    for data_dict in tqdm(val_loader):
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        gt, gt_mask, gt_bbox, gt_pres_mask = get_input(params, data_dict)
        B = gt.shape[0]

        # apply blackout to to input frames
        if True:
            blackout_mask_orig = (torch.rand(gt.shape[0:2]) < blackout_p).float()
        else:
            blackout_mask_orig = torch.zeros(gt.shape[0:2])
            # set every 4th frame to 1
            for i in range(0, blackout_mask_orig.shape[1], 4):
                blackout_mask_orig[:, i] = 1
        zero_mask = torch.zeros((B,6)).float()
        blackout_mask = torch.cat((zero_mask, blackout_mask_orig), dim=1).to(gt.device)
        data_dict['img'] = data_dict['img'] * (1-blackout_mask[:, :, None, None, None])

        # take model output
        out_dict = model(data_dict)
        pred, pred_mask, pred_bbox = get_output(params, out_dict)

        for b in range(B):
            # compute metrics
            metric_dict = pred_eval_step(
                gt=gt[b:b+1],
                pred=pred[b:b+1],
                lpips_fn=loss_fn_vgg,
                gt_mask=gt_mask[b:b+1],
                pred_mask=pred_mask[b:b+1],
                gt_pres_mask=gt_pres_mask[b:b+1],
                gt_bbox=gt_bbox[b:b+1],
                pred_bbox=pred_bbox[b:b+1],
                eval_traj=True,
            )
            metric_dict['blackout'] = blackout_mask_orig[b].tolist()
            metric_complete = append_statistics(metric_dict, metric_complete)

            if len(video_dic['gt']) < num_videos:
                video_dic['gt'] = video_dic['gt'] + [gt[b].detach().cpu()]
                video_dic['pred'] = video_dic['pred'] + [pred[b].detach().cpu()]
                video_dic['input'] = video_dic['input'] + [data_dict['img'][b].detach().cpu()]
                video_dic['blackout'] = video_dic['blackout'] + [blackout_mask_orig[b].detach().cpu()]

                if len(video_dic['gt']) == num_videos:
                    print('saving videos')
                    with open(os.path.join(f'statistics', f'video_dic.pkl'), 'wb') as f:
                        pickle.dump(video_dic, f)

    # compute average metrics
    average_dic = {}
    for key in metric_complete:

        # take average over all frames
        average_dic[key + 'complete_average'] = np.mean(metric_complete[key])
        average_dic[key + 'complete_std']     = np.std(metric_complete[key])
        print(f'{key} complete average: {average_dic[key + "complete_average"]:.4f} +/- {average_dic[key + "complete_std"]:.4f}')

        if evaluation_mode == 'blackout':
            # take average only for frames where blackout occurs
            blackout_mask = np.array(metric_complete['blackout']) > 0
            average_dic[key + 'blackout_average'] = np.mean(np.array(metric_complete[key])[blackout_mask])
            average_dic[key + 'blackout_std']     = np.std(np.array(metric_complete[key])[blackout_mask])
            average_dic[key + 'visible_average']  = np.mean(np.array(metric_complete[key])[blackout_mask == False])
            average_dic[key + 'visible_std']      = np.std(np.array(metric_complete[key])[blackout_mask == False])

            print(f'{key} blackout average: {average_dic[key + "blackout_average"]:.4f} +/- {average_dic[key + "blackout_std"]:.4f}')
            print(f'{key} visible average: {average_dic[key + "visible_average"]:.4f} +/- {average_dic[key + "visible_std"]:.4f}')

    with open(os.path.join(f'statistics', f'{evaluation_mode}_metric_complete.pkl'), 'wb') as f:
        pickle.dump(metric_complete, f)
    with open(os.path.join(f'statistics', f'{evaluation_mode}_metric_average.pkl'), 'wb') as f:
        pickle.dump(average_dic, f)

def main():
    params.ddp = False
    params.n_sample_frames = 42 + 6
    params.input_frames = 6
    params.val_batch_size = 4
    model = build_model(params)
    model.load_state_dict(
        torch.load(args.weight, map_location='cpu')['state_dict'])
    model = torch.nn.DataParallel(model).cuda().eval()

    test_blackout(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract slots from videos')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--subset', type=str, default='training')  # Physion
    parser.add_argument(
        '--weight', type=str, required=True, help='pretrained model weight')
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,  # './data/CLEVRER/slots.pkl'
        help='path to save slots',
    )
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    if 'physion' in args.params:
        params.dataset = f'physion_{args.subset}'
    assert params.dataset in args.save_path

    torch.backends.cudnn.benchmark = True
    main()
