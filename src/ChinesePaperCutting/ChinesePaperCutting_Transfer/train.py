import os
import argparse
import csv
import re
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from model.configuration import TransModule_Config
from model.framework import Encoder
from net import vgg, TransModule, Decoder_MVGG, Net
from dataset_sampler import SimpleDataset, InfiniteSamplerWrapper
from scheduler import CosineAnnealingWarmUpLR
from tools import LossImprovementTracker, load_network_weights, save_checkpoint


"""Parameters that needs attention
1.epoch           : How many iterations this training has
2.epoch_start        : From which iteration to start the training
3.checkpoint_save_interval : The interval to save checkpoints
4.loss_count_interval    : The interval to calculate the average loss
"""

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg_dir', type=str, default='./pre_trained_models/vgg_normalised.pth')

# training options
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epoch', type=int, default=40000)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--seed', type=int, default=2026,
                    help='Random seed for Python, NumPy, PyTorch, CUDA, and samplers')

parser.add_argument('--content_weight', type=float, default=2)
parser.add_argument('--style_weight', type=float, default=3)
parser.add_argument('--id1_weight', type=float, default=50)
parser.add_argument('--id2_weight', type=float, default=1)
parser.add_argument('--use_pcp_loss', action='store_true',
                    help='Enable the paper-cut edge/frequency prior loss')
parser.add_argument('--pcp_weight', type=float, default=0.0,
                    help='Weight for the paper-cut prior loss')
parser.add_argument('--pcp_warmup_steps', type=int, default=0,
                    help='Linearly ramp pcp_weight from 0 to its target over this many steps; 0 disables warmup')
parser.add_argument('--pcp_edge_weight', type=float, default=1.0,
                    help='Edge component weight inside the paper-cut prior loss')
parser.add_argument('--pcp_freq_weight', type=float, default=0.1,
                    help='Frequency component weight inside the paper-cut prior loss')
parser.add_argument('--pcp_palette_weight', type=float, default=0.0,
                    help='Palette component weight inside the paper-cut prior loss')
parser.add_argument('--pcp_smooth_weight', type=float, default=0.0,
                    help='Smoothness component weight inside the paper-cut prior loss')
parser.add_argument('--pcp_texture_weight', type=float, default=0.0,
                    help='High-frequency texture component weight inside the paper-cut prior loss')
parser.add_argument('--ppeb_mode', type=str, default='full',
                    choices=['off', 'edge', 'freq', 'full'],
                    help='PPEB ablation mode')
parser.add_argument('--use_style_adapter', action='store_true',
                    help='Enable the style-conditioned texture adapter after the transfer module')
parser.add_argument('--style_adapter_alpha', type=float, default=0.1,
                    help='Residual scale for the style-conditioned texture adapter')

# save and count options
parser.add_argument('--checkpoint_save_interval', type=int, default=10000)
parser.add_argument('--loss_count_interval', type=int, default=400)
parser.add_argument('--latest_checkpoint_interval', type=int, default=0,
                    help='Interval to overwrite latest_checkpoint.pkl; 0 disables it')
parser.add_argument('--best_checkpoint_metric', type=str, default='log_avg',
                    choices=['step', 'log_avg'],
                    help='Metric used to save best_checkpoint.pkl: raw step loss or logged average loss')
parser.add_argument('--auto_resume', action='store_true',
                    help='Resume from latest checkpoint in checkpoint_save_path when available')
parser.add_argument('--early_stop_patience', type=int, default=0,
                    help='Number of loss reports without improvement before stopping; 0 disables early stopping')
parser.add_argument('--early_stop_min_delta', type=float, default=0.1)
parser.add_argument('--early_stop_smoothing', type=float, default=0.01,
                    help='EMA smoothing factor for the per-step loss used by early stopping; 0 uses raw step loss')
parser.add_argument('--early_stop_warmup', type=int, default=0,
                    help='Do not early stop before this training step')
parser.add_argument('--reset_early_stop_on_resume', action='store_true',
                    help='Ignore early-stop monitor state from the resume checkpoint')
parser.add_argument('--resume_train', type=bool, default=False, help='Use checkpoints to train or not ')
parser.add_argument('--checkpoint_save_path', type=str, default='./pre_trained_models/checkpoint1',
                    help='Directory path to save a checkpoint')
parser.add_argument('--checkpoint_import_path', type=str, default='',
                    help='Directory path to the importing checkpoint')
parser.add_argument('--init_checkpoint_path', type=str, default='',
                    help='Model-only warm-start checkpoint path; does not load optimizer, scheduler, epoch, or logs')

args = parser.parse_args()


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


set_random_seed(args.seed)

# Print args
print('Running args: ')
for k, v in sorted(vars(args).items()):
    print(k, '=', v)
print()

if not os.path.exists(args.checkpoint_save_path):
    os.mkdir(args.checkpoint_save_path)

epoch_start = 0
loss_count_interval = args.loss_count_interval
EXTRA_LOSS_KEYS = [
  'loss_cut_attn_original',
  'loss_pcp_total',
  'loss_pcp_edge',
  'loss_pcp_freq',
  'loss_pcp_palette',
  'loss_pcp_smooth',
  'loss_pcp_texture',
  'loss_pcp_weighted',
]
EXTRA_LOG_KEYS = ['log_' + key.removeprefix('loss_') for key in EXTRA_LOSS_KEYS]


def build_extra_logs(logs_by_key):
  return {log_key: logs_by_key[loss_key] for loss_key, log_key in zip(EXTRA_LOSS_KEYS, EXTRA_LOG_KEYS)}


def build_checkpoint_extra_state(**kwargs):
  state = {'seed': args.seed}
  state.update(kwargs)
  return state


def best_metric_name():
  if args.best_checkpoint_metric == 'step':
    return 'loss_all_step'
  return 'loss_all_log_avg'


def current_pcp_weight(step):
  if not args.use_pcp_loss:
    return 0.0
  if args.pcp_warmup_steps <= 0:
    return args.pcp_weight
  return args.pcp_weight * min(1.0, step / float(args.pcp_warmup_steps))


def write_loss_history_csv(checkpoint_dir, steps, log_c, log_s, log_id1, log_id2, log_all, logs_by_key):
  path = os.path.join(checkpoint_dir, 'loss_history.csv')
  fieldnames = [
    'step',
    'loss_all',
    'loss_content',
    'loss_style',
    'loss_id1',
    'loss_id2',
  ] + EXTRA_LOSS_KEYS
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for idx, step in enumerate(steps):
      row = {
        'step': step,
        'loss_all': log_all[idx],
        'loss_content': log_c[idx],
        'loss_style': log_s[idx],
        'loss_id1': log_id1[idx],
        'loss_id2': log_id2[idx],
      }
      for loss_key in EXTRA_LOSS_KEYS:
        values = logs_by_key.get(loss_key, [])
        row[loss_key] = values[idx] if idx < len(values) else ''
      writer.writerow(row)


def find_latest_checkpoint(checkpoint_dir):
  latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pkl')
  if os.path.exists(latest_path):
    return latest_path

  best_epoch = -1
  best_path = ''
  pattern = re.compile(r'checkpoint_(\d+)_epoch\.pkl$')
  for name in os.listdir(checkpoint_dir):
    match = pattern.match(name)
    if match:
      epoch = int(match.group(1))
      if epoch > best_epoch:
        best_epoch = epoch
        best_path = os.path.join(checkpoint_dir, name)
  return best_path


# Model Config
transModule_config = TransModule_Config(
            nlayer=3,
            d_model=768,
            nhead=8,
            mlp_ratio=4,
            qkv_bias=False,
            attn_drop=0.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            norm_first=True
            )

# Datasets
train_transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
])
dataset_content = SimpleDataset(args.content_dir, transforms=train_transform)
dataset_style = SimpleDataset(args.style_dir, transforms=train_transform)
sampler_content = InfiniteSamplerWrapper(dataset_content, seed=args.seed)
sampler_style = InfiniteSamplerWrapper(dataset_style, seed=args.seed + 1)
dataloader_content_iter = iter(DataLoader(dataset_content,
                      batch_size=args.batch_size,
                      sampler=sampler_content,
                      num_workers=0))
dataloader_style_iter = iter(DataLoader(dataset_style,
                      batch_size=args.batch_size,
                      sampler=sampler_style,
                      num_workers=0))


# Hardware Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
   
# Models
vgg.load_state_dict(torch.load(args.vgg_dir))
encoder = Encoder(
  img_size=224,
  patch_size=2,
  in_chans=3,
  embed_dim=192,
  depths=[2, 2, 2],
  nhead=[3, 6, 12],
  strip_width=[2, 4, 7],
  drop_path_rate=0.,
  patch_norm=True
)
decoder = Decoder_MVGG(d_model=768, seq_input=True)
transModule = TransModule(
  transModule_config,
  use_style_adapter=args.use_style_adapter,
  style_adapter_alpha=args.style_adapter_alpha,
)

network = Net(encoder, decoder, transModule, vgg)
network.use_paper_cut_prior_loss = args.use_pcp_loss
network.paper_cut_prior_weight = args.pcp_weight
network.paper_cut_prior_loss.edge_weight = args.pcp_edge_weight
network.paper_cut_prior_loss.freq_weight = args.pcp_freq_weight
network.paper_cut_prior_loss.palette_weight = args.pcp_palette_weight
network.paper_cut_prior_loss.smooth_weight = args.pcp_smooth_weight
network.paper_cut_prior_loss.texture_weight = args.pcp_texture_weight
for module in network.modules():
    if hasattr(module, 'set_mode'):
        module.set_mode(args.ppeb_mode)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': network.encoder.parameters()},
    {'params': network.decoder.parameters()},
    {'params': network.transModule.parameters()},
], lr=args.base_lr)
scheduler = CosineAnnealingWarmUpLR(optimizer, warmup_step=args.epoch//4, max_step=args.epoch, min_lr=0)


# Whether to use parameters from checkpoints
if args.auto_resume and not args.resume_train:
  latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_save_path)
  if latest_checkpoint_path:
    args.resume_train = True
    args.checkpoint_import_path = latest_checkpoint_path

if args.resume_train:
  print('loading checkpoint...')
  checkpoint = torch.load(args.checkpoint_import_path, map_location=device, weights_only=False)

  load_network_weights(
    network,
    checkpoint,
    allow_partial_transmodule=args.use_style_adapter,
  )
    
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])

  for state in optimizer.state.values():
    for k, v in state.items():
      if torch.is_tensor(v):
        state[k] = v.to(device)

  log_c = checkpoint['log_c']
  log_s = checkpoint['log_s']
  log_id1 = checkpoint['log_id1']
  log_id2 = checkpoint['log_id2']
  log_all = checkpoint['log_all']
  logs_by_key = {}
  for loss_key, log_key in zip(EXTRA_LOSS_KEYS, EXTRA_LOG_KEYS):
    logs_by_key[loss_key] = checkpoint.get(log_key, [])

  epoch_start = checkpoint['epoch']
  checkpoint_loss_count_interval = checkpoint['loss_count_interval']
  loss_count_interval = args.loss_count_interval
  log_steps = checkpoint.get('log_steps')
  if log_steps is None:
    log_steps = [checkpoint_loss_count_interval * (idx + 1) for idx in range(len(log_all))]
  print('loading finished from {}'.format(args.checkpoint_import_path))
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
else:
  if args.init_checkpoint_path:
    print('initializing model weights from {}'.format(args.init_checkpoint_path))
    checkpoint = torch.load(args.init_checkpoint_path, map_location=device, weights_only=False)
    load_network_weights(
      network,
      checkpoint,
      allow_partial_transmodule=args.use_style_adapter,
    )
    print('model weight initialization finished')
  log_c, log_s, log_id1, log_id2, log_all = [],[],[],[],[]
  logs_by_key = {loss_key: [] for loss_key in EXTRA_LOSS_KEYS}
  log_steps = []

log_c_temp, log_s_temp, log_id1_temp, log_id2_temp, log_all_temp = [],[],[],[],[]
logs_temp_by_key = {loss_key: [] for loss_key in EXTRA_LOSS_KEYS}
checkpoint_best_loss = None
checkpoint_best_loss_step = None
checkpoint_early_stop_best_loss = None
checkpoint_early_stop_monitor_loss = None
checkpoint_early_stop_bad_steps = 0
if args.resume_train:
  if checkpoint.get('best_metric_name') == best_metric_name():
    checkpoint_best_loss = checkpoint.get('best_metric_value')
    checkpoint_best_loss_step = checkpoint.get('best_metric_step')
  checkpoint_early_stop_best_loss = checkpoint.get('early_stop_best_metric_value', checkpoint_best_loss)
  checkpoint_early_stop_monitor_loss = checkpoint.get('early_stop_monitor_metric_value')
  checkpoint_early_stop_bad_steps = checkpoint.get('early_stop_bad_steps', 0)
  if args.reset_early_stop_on_resume:
    checkpoint_early_stop_best_loss = None
    checkpoint_early_stop_monitor_loss = None
    checkpoint_early_stop_bad_steps = 0
  if args.early_stop_smoothing > 0.0 and checkpoint_early_stop_monitor_loss is None:
    checkpoint_early_stop_best_loss = None
    checkpoint_early_stop_bad_steps = 0
if checkpoint_best_loss is None and log_all:
  checkpoint_best_loss = min(log_all)
  checkpoint_best_loss_step = log_steps[log_all.index(checkpoint_best_loss)]
if checkpoint_early_stop_best_loss is None and not (
  args.early_stop_smoothing > 0.0 and checkpoint_early_stop_monitor_loss is None
):
  checkpoint_early_stop_best_loss = checkpoint_best_loss
loss_tracker = LossImprovementTracker(
  best_loss=checkpoint_best_loss,
  best_step=checkpoint_best_loss_step,
  early_stop_best_loss=checkpoint_early_stop_best_loss,
  early_stop_min_delta=args.early_stop_min_delta,
  early_stop_patience=args.early_stop_patience,
  early_stop_warmup=args.early_stop_warmup,
  bad_steps=checkpoint_early_stop_bad_steps,
  early_stop_smoothing=args.early_stop_smoothing,
  early_stop_monitor_loss=checkpoint_early_stop_monitor_loss,
)


# Load the model to device
network.to(device)


# Training
if __name__ == '__main__':
  progress = tqdm(
    range(epoch_start + 1, args.epoch + 1),
    total=args.epoch,
    initial=epoch_start,
    desc='train',
    unit='step',
    dynamic_ncols=True,
  )
  for i in progress:
    
    # data samples
    i_c = next(dataloader_content_iter).to(device)
    i_s = next(dataloader_style_iter).to(device)

    # calculate losses
    network.paper_cut_prior_weight = current_pcp_weight(i)
    loss_c, loss_s, loss_id_1, loss_id_2,loss_cut, _ = network(i_c, i_s)
    loss_all = args.content_weight*loss_c + args.style_weight*loss_s + args.id1_weight*loss_id_1 + args.id2_weight*loss_id_2 +loss_cut
    current_loss = loss_all.item()
    current_step = i
    
    log_c_temp.append(loss_c.item())
    log_s_temp.append(loss_s.item())
    log_id1_temp.append(loss_id_1.item())
    log_id2_temp.append(loss_id_2.item())
    log_all_temp.append(loss_all.item())
    for loss_key in EXTRA_LOSS_KEYS:
      value = network.latest_loss_terms.get(loss_key)
      if value is None:
        value = loss_all.new_tensor(0.0)
      logs_temp_by_key[loss_key].append(value.item())

    monitor_result = loss_tracker.update(current_loss=current_loss, step=current_step)
    if args.best_checkpoint_metric == 'step':
      checkpoint_best_loss = monitor_result.best_loss
      checkpoint_best_loss_step = monitor_result.best_step
    if args.best_checkpoint_metric == 'step' and monitor_result.is_best:
      save_checkpoint(
        encoder=network.encoder,
        transModule=network.transModule,
        decoder=network.decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=i,
        log_c=log_c,
        log_s=log_s,
        log_id1=log_id1,
        log_id2=log_id2,
        log_all=log_all,
        loss_count_interval=loss_count_interval,
        save_path=os.path.join(args.checkpoint_save_path, 'best_checkpoint.pkl'),
        extra_logs=build_extra_logs(logs_by_key),
        extra_state=build_checkpoint_extra_state(
          best_metric_name=best_metric_name(),
          best_metric_value=checkpoint_best_loss,
          best_metric_step=checkpoint_best_loss_step,
          early_stop_best_metric_value=monitor_result.early_stop_best_loss,
          early_stop_monitor_metric_value=monitor_result.early_stop_monitor_loss,
          early_stop_bad_steps=monitor_result.bad_steps,
          log_steps=log_steps,
        )
      )
      print('Saved best checkpoint at step {:d}: loss_all_step={:.6f}'.format(i, checkpoint_best_loss))

    if args.early_stop_patience > 0 and i >= args.early_stop_warmup and not monitor_result.is_early_stop_best:
      print('Early stop step counter: {:d}/{:d}, early_stop_monitor_loss={:.6f}, early_stop_best_loss={:.6f}'.format(
        monitor_result.bad_steps,
        args.early_stop_patience,
        monitor_result.early_stop_monitor_loss,
        monitor_result.early_stop_best_loss
      ))
    if monitor_result.should_stop:
      save_checkpoint(
        encoder=network.encoder,
        transModule=network.transModule,
        decoder=network.decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=i,
        log_c=log_c,
        log_s=log_s,
        log_id1=log_id1,
        log_id2=log_id2,
        log_all=log_all,
        loss_count_interval=loss_count_interval,
        save_path=os.path.join(args.checkpoint_save_path, 'early_stop_checkpoint.pkl'),
        extra_logs=build_extra_logs(logs_by_key),
        extra_state=build_checkpoint_extra_state(
          best_metric_name=best_metric_name(),
          best_metric_value=checkpoint_best_loss,
          best_metric_step=checkpoint_best_loss_step,
          early_stop_best_metric_value=monitor_result.early_stop_best_loss,
          early_stop_monitor_metric_value=monitor_result.early_stop_monitor_loss,
          early_stop_bad_steps=monitor_result.bad_steps,
          log_steps=log_steps,
        )
      )
      print('Early stopping at step {:d}'.format(i))
      break

    # update parameters
    optimizer.zero_grad()
    loss_all.backward()
    optimizer.step()
    scheduler.step()

    # calculate average loss
    if i % loss_count_interval == 0:
      log_c.append(np.mean(np.array(log_c_temp)))
      log_s.append(np.mean(np.array(log_s_temp)))
      log_id1.append(np.mean(np.array(log_id1_temp)))
      log_id2.append(np.mean(np.array(log_id2_temp)))
      log_all.append(np.mean(np.array(log_all_temp)))
      for loss_key in EXTRA_LOSS_KEYS:
        logs_by_key[loss_key].append(np.mean(np.array(logs_temp_by_key[loss_key])))
      log_steps.append(i)

      print('Epoch {:d}: '.format(i) + str(log_all[-1]))
      print('Loss details: ' + ', '.join(
        '{}={:.6f}'.format(loss_key, logs_by_key[loss_key][-1])
        for loss_key in EXTRA_LOSS_KEYS
      ))
      write_loss_history_csv(
        args.checkpoint_save_path,
        log_steps,
        log_c,
        log_s,
        log_id1,
        log_id2,
        log_all,
        logs_by_key,
      )
      current_loss = log_all[-1]
      current_step = i
      progress.set_postfix(
        loss='{:.4f}'.format(current_loss),
        content='{:.4f}'.format(log_c[-1]),
        style='{:.4f}'.format(log_s[-1]),
        pcp='{:.4f}'.format(logs_by_key['loss_pcp_total'][-1]),
        pcp_w='{:.3f}'.format(network.paper_cut_prior_weight),
        early_stop='{}/{}'.format(monitor_result.bad_steps, args.early_stop_patience),
      )

      if args.best_checkpoint_metric == 'log_avg':
        log_avg_loss = float(log_all[-1])
        if checkpoint_best_loss is None or log_avg_loss < checkpoint_best_loss:
          checkpoint_best_loss = log_avg_loss
          checkpoint_best_loss_step = i
          save_checkpoint(
            encoder=network.encoder,
            transModule=network.transModule,
            decoder=network.decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=i,
            log_c=log_c,
            log_s=log_s,
            log_id1=log_id1,
            log_id2=log_id2,
            log_all=log_all,
            loss_count_interval=loss_count_interval,
            save_path=os.path.join(args.checkpoint_save_path, 'best_checkpoint.pkl'),
            extra_logs=build_extra_logs(logs_by_key),
            extra_state=build_checkpoint_extra_state(
              best_metric_name=best_metric_name(),
              best_metric_value=checkpoint_best_loss,
              best_metric_step=checkpoint_best_loss_step,
              early_stop_best_metric_value=monitor_result.early_stop_best_loss,
              early_stop_monitor_metric_value=monitor_result.early_stop_monitor_loss,
              early_stop_bad_steps=monitor_result.bad_steps,
              log_steps=log_steps,
            )
          )
          print('Saved best checkpoint at step {:d}: loss_all_log_avg={:.6f}'.format(i, checkpoint_best_loss))

      log_c_temp, log_s_temp = [],[]
      log_id1_temp, log_id2_temp = [],[]
      log_all_temp = []
      logs_temp_by_key = {loss_key: [] for loss_key in EXTRA_LOSS_KEYS}

    if args.latest_checkpoint_interval > 0 and i % args.latest_checkpoint_interval == 0:
      save_checkpoint(
        encoder=network.encoder,
        transModule=network.transModule,
        decoder=network.decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=i,
        log_c=log_c,
        log_s=log_s,
        log_id1=log_id1,
        log_id2=log_id2,
        log_all=log_all,
        loss_count_interval=loss_count_interval,
        save_path=os.path.join(args.checkpoint_save_path, 'latest_checkpoint.pkl'),
        extra_logs=build_extra_logs(logs_by_key),
        extra_state=build_checkpoint_extra_state(
          best_metric_name=best_metric_name(),
          best_metric_value=checkpoint_best_loss,
          best_metric_step=checkpoint_best_loss_step,
          early_stop_best_metric_value=loss_tracker.early_stop_best_loss,
          early_stop_monitor_metric_value=loss_tracker.early_stop_monitor_loss,
          early_stop_bad_steps=loss_tracker.bad_steps,
          log_steps=log_steps,
        )
      )

    # save a checkpoint
    if i % args.checkpoint_save_interval == 0:
      save_checkpoint(
        encoder=network.encoder,
        transModule=network.transModule,
        decoder=network.decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=i,
        log_c=log_c,
        log_s=log_s,
        log_id1=log_id1,
        log_id2=log_id2,
        log_all=log_all,
        loss_count_interval=loss_count_interval,
        save_path=os.path.join(args.checkpoint_save_path, 'checkpoint_{}_epoch.pkl'.format(i)),
        extra_logs=build_extra_logs(logs_by_key),
        extra_state=build_checkpoint_extra_state(
          best_metric_name=best_metric_name(),
          best_metric_value=checkpoint_best_loss,
          best_metric_step=checkpoint_best_loss_step,
          early_stop_best_metric_value=loss_tracker.early_stop_best_loss,
          early_stop_monitor_metric_value=loss_tracker.early_stop_monitor_loss,
          early_stop_bad_steps=loss_tracker.bad_steps,
          log_steps=log_steps,
        )
      )
