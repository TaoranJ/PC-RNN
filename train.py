# -*- coding: utf-8 -*-

import os
import argparse
import time
import math

import torch
import torch.optim as optimizer

from data import load_dataset, collate_fn
from models import Encoder, Decoder, PCRNN

pparser = argparse.ArgumentParser()
# =============================================================================
# =================================== Data ====================================
# =============================================================================
pparser.add_argument('--ob-ratio', type=float, default=.5,
                     help='observation window')
pparser.add_argument('--use-cuda', type=str, choices=['0', '1'], default='0',
                     help='cuda device number')
pparser.add_argument('--use-category', action='store_true',
                     help='use category')
# =============================================================================
# ================================== Model ====================================
# =============================================================================
pparser.add_argument('--embed-dim', type=int, default=16,
                     help='Embedding dimension')
pparser.add_argument('--pencoder-hidden-dim', type=int, default=32,
                     help='Hidden dim for patent encoder.')
pparser.add_argument('--oencoder-hidden-dim', type=int, default=16,
                     help='Hidden dim for inventor/assignee encoder.')
pparser.add_argument('--decoder-hidden-dim', type=int, default=32,
                     help='Hidden dim for decoder.')
pparser.add_argument('--decoder-inner-dim', type=int, default=64,
                     help='Inner dim for decoder.')
# =============================================================================
# ================================ Optimizer ==================================
# =============================================================================
pparser.add_argument('--lr', type=float, default=.001, help='Learning rate')
pparser.add_argument('--weight-decay', type=float, default=0,
                     help='Weight decay.')
pparser.add_argument('--clip', type=float, default=50.0,
                     help='Weight clipping')
# =============================================================================
# ================================= Training ==================================
# =============================================================================
pparser.add_argument('--epochs', type=int, default=10, help='Epochs')
pparser.add_argument('--batch-size', type=int, default=256,
                     help='Minibatch size')
pparser.add_argument('--checkpoint-path', type=str, default='checkpoint.pth',
                     help='Checkpoint path.')
pparser.add_argument('--tune-lr', action='store_true', help='tune lr?')
# =============================================================================
# ================================ Evaluation =================================
# =============================================================================
pparser.add_argument('--use-best', action='store_true',
                     help='Use best optimiztion point.')
pparser.add_argument('--eval-train', action='store_true',
                     help='Evaluation on training set.')
pparser.add_argument('--eval-test', action='store_true',
                     help='Evaluation on test set.')
args = pparser.parse_args()
args.pad = 0
device = torch.device('cuda:' + args.use_cuda)
args.device = device
print(args)


def generate_checkpoint_path():
    """Generate checkpoint path."""

    cuda = 'cuda' + args.use_cuda
    cat = 'cat' if args.use_category else 'sub-cat'
    args.checkpoint_path += '.{}.{}.ob{}.bs{}.pth'.format(
            cuda, cat, str(int(args.ob_ratio * 10)),  str(args.batch_size))


# =============================================================================
# =============================== Load dataset ================================
# =============================================================================
train_dataset, test_dataset, num_categories = load_dataset(args)
args.num_categories = num_categories + 1  # add category 0 for padding
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=True, drop_last=False)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=True, drop_last=False)
args.train_len = sum([1 for batch in train_loader])  # len of training set
# =============================================================================
# =========================== Model initialization ============================
# =============================================================================
encoder = Encoder(num_categories=args.num_categories,
                  embed_dim=args.embed_dim,
                  p_encoder_hidden_dim=args.pencoder_hidden_dim,
                  o_encoder_hidden_dim=args.oencoder_hidden_dim)
decoder = Decoder(num_categories=args.num_categories,
                  embed_dim=args.embed_dim,
                  p_encoder_hidden_dim=args.pencoder_hidden_dim,
                  o_encoder_hidden_dim=args.oencoder_hidden_dim,
                  p_decoder_hidden_dim=args.decoder_hidden_dim,
                  p_decoder_inner_dim=args.decoder_inner_dim)
model = PCRNN(encoder, decoder).to(device)
# =============================================================================
# ========================= Optimizer initialization ==========================
# =============================================================================
optim = optimizer.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
optim_scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(
        optim, patience=10, factor=.5, min_lr=.0005)
generate_checkpoint_path()


# =============================================================================
# =============================== Train model =================================
# =============================================================================
def unzip_minibatch(data):
    """Unzip minibatch and load data to device.

    Parameters
    ----------
    data : list
        A minibatch of data.

    Returns
    -------
    patent_src : dict
        Minibatch data used on source side.
    patent_tgt : dict
        Minibatch data used on target side.
    assignee : dict
        Minibatch data for assignee series.
    inventor : dict
        Minibatch data for inventor series.

    """

    src_pts, tgt_pts, src_pcat, tgt_pcat, length, mask, ats, aorg_idx, \
        alength, its, iorg_idx, ilength = data
    patent_src = {'pts': src_pts.to(device), 'pcat': src_pcat.to(device),
                  'length': length.to(device)}
    patent_tgt = {'pts': tgt_pts.to(device), 'pcat': tgt_pcat.to(device),
                  'mask': mask.to(device)}
    assignee = {'ts': None, 'org_idx': None, 'length': None}
    inventor = {'ts': None, 'org_idx': None, 'length': None}
    assignee['ts'], assignee['org_idx'] = ats.to(device), aorg_idx
    assignee['length'] = alength.to(device)
    inventor['ts'], inventor['org_idx'] = its.to(device), iorg_idx
    inventor['length'] = ilength.to(device)
    return patent_src, patent_tgt, assignee, inventor


def cal_loss(tgt_ts_output, tgt_cat_output, patent_tgt):
    """Calculate loss for the forward propagation.

    Parameters
    ----------
    tgt_ts_output : :class:`torch.Tensor`
        Timestamp predictions on target side.
    tgt_cat_output : :class:`torch.Tensor`
        Category predictions on target side.
    patent_tgt : dict
        Minibatch data used on target side.

    Returns
    -------
        loss.

    """

    # loss for timestamp prediction
    ts_loss = patent_tgt['pts'] - tgt_ts_output.squeeze(-1)
    ts_loss = torch.abs(ts_loss).masked_select(patent_tgt['mask']).sum()
    # loss for category prediction
    cat_loss = sum(NLLLoss_mask(p, t, m)
                   for p, t, m in zip(tgt_cat_output, patent_tgt['pcat'],
                                      patent_tgt['mask']))
    return ts_loss + cat_loss


def NLLLoss_mask(pred, target, mask):
    """

    Customized NLLLoss for masked sequences. Losses are only calculated on the
    non-pad targets which are masked out by the mask.

    Parameters
    ----------
    pred : :class:`torch.Tensor`
        Category prediction for each element in the minibatch, tensor of shape
        (batch, num_categories). This should be the output of a softmax layer.
    target : :class:`torch.Tensor`
        True categories for each element in this minibatch, tensor of shape
        (batch)
    mask : :class:`torch.Tensor`
        Mask out non-pad position in the target, tensor of shape (batch).

    Returns
    -------
    loss : float
        Loss of this minibatch.

    """

    pred = torch.gather(pred, 1, target.view(-1, 1))
    cross_entropy = - pred.squeeze(1)
    loss = cross_entropy.masked_select(mask).sum()
    return loss


def train_step(model, optim, data):
    """One training step.

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        PCRNN.
    optim : :class:`torch.optim.Optimizer`
        Optimizer for PCRNN.
    data : list
        A minibatch of data.

    """

    optim.zero_grad()
    patent_src, patent_tgt, assignee, inventor = unzip_minibatch(data)
    tgt_ts_output, tgt_cat_output = model(patent_src, assignee, inventor,
                                          patent_tgt)
    loss = cal_loss(tgt_ts_output, tgt_cat_output, patent_tgt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optim.step()
    return loss.item()


def time_since(since, m_padding=2, s_padding=2):
    """Elapsed time since last record point."""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m:{}s'.format(str(int(m)).zfill(m_padding),
                            str(int(s)).zfill(s_padding))


def train(model, optim, dataloader, optim_scheduler):
    """Training.

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        PCRNN.
    optim : :class:`torch.optim.Optimizer`
        Optimizer for the model.
    dataloader : :class:`torch.utils.data.DataLoader`
        Dataloader for training set.

    """

    model.train()
    start_epoch, best_epoch_loss, epoch_loss = time.time(), 1e15, 0
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        for batch in dataloader:
            loss = train_step(model, optim, batch)
            epoch_loss += loss
        if args.tune_lr:
            optim_scheduler.step(epoch_loss)
        print('[Epochs: {:02d}/{:02d}], Elapsed time: {} '
              'Loss: {:.4f}'.format(epoch, args.epochs,
                                    time_since(start_epoch), epoch_loss))
        if epoch_loss <= best_epoch_loss:
            torch.save({'model': encoder.state_dict()},
                       args.checkpoint_path + '.best')
            best_epoch_loss = epoch_loss
        epoch_losses.append(epoch_loss)
        epoch_loss = 0
    return model


# =============================================================================
# ============================== Evaluate model ===============================
# =============================================================================

def collect_results(tgt_ts_output, tgt_cat_output, patent_tgt):
    """Prepare results and ground truth for evaluation.

    Parameters
    ----------
    tgt_ts_output : :class:`torch.Tensor`
        Prediction for arrival time, a tensor of shape (seq_len, batch, 1).
    tgt_cat_output: :class:`torch.Tensor`
        Prediction for category, a tensor of shape (seq_len, batch,
        num_categories).
    patent_tgt : dict
        Ground truth data for patent prediction including real timestamp,
        category and mask.

    Returns
    -------
    mae : :class:`torch.Tensor`
        (loss, # of points)
    acc : :class:`torch.Tensor`
        (# of correct predictions, # of points)

    """

    # get predictions and ground ready. y'all agree on dimensions first.
    tgt_pts = patent_tgt['pts'].unsqueeze(-1)
    tgt_pcat = patent_tgt['pcat'].unsqueeze(-1)
    mask = patent_tgt['mask'].unsqueeze(-1)
    pred_cat = tgt_cat_output.topk(1, dim=2)[1]
    assert(tgt_pts.size() == tgt_pcat.size() == mask.size() == pred_cat.size()
           == tgt_ts_output.size())
    # timestamp predictions and ground truth
    pred_ts = tgt_ts_output.masked_select(mask)
    tgt_ts = tgt_pts.masked_select(mask)
    # category predictions and ground truth
    pred_cat = pred_cat.masked_select(mask)
    tgt_cat = tgt_pcat.masked_select(mask)
    # calculate mae and accuracy
    mae = torch.tensor([torch.abs(pred_ts - tgt_ts).sum().item(),
                        pred_ts.size(0)])
    acc = torch.tensor([torch.sum((pred_cat == tgt_cat)).item(),
                        pred_cat.size(0)], dtype=torch.float)
    return mae, acc


def evaluate_step(model, data):
    """Evaluation on one minibatch.

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        PCRNN.
    data : dict
        One minibatch of data.

    Returns
    -------
    mae : :class:`torch.Tensor`
        (loss, # of points)
    acc : :class:`torch.Tensor`
        (# of correct predictions, # of points)

    """

    patent_src, patent_tgt, assignee, inventor = unzip_minibatch(data)
    tgt_ts_output, tgt_cat_output = model(patent_src, assignee, inventor,
                                          patent_tgt)
    mae, acc = collect_results(tgt_ts_output, tgt_cat_output, patent_tgt)
    return mae, acc


def evaluate(model, dataloader):
    """Calculate mean absolute value and accuracy.

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        PCRNN.
    dataloader : :class:`torch.utils.data.DataLoader`
        Dataloader for dataset to evaluate.

    """

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    mae, acc = torch.tensor([0., 0.]), torch.tensor([0., 0.])
    with torch.no_grad():
        for batch in dataloader:
            mae_step, acc_step = evaluate_step(model, batch)
            mae += mae_step
            acc += acc_step
        with open(args.checkpoint_path[:-4] + '.eval.txt', 'a') as ofp:
            print('MAE: {:.4f}, ACC: {:.4f}'.format(
                mae[0] / mae[1], acc[0] / acc[1]), file=ofp)


# =============================================================================
# ================================ Main entry =================================
# =============================================================================

if os.path.exists(args.checkpoint_path):  # continue previous training
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
model = train(model, optim, train_loader, optim_scheduler)
torch.save({'model': model.state_dict()}, args.checkpoint_path)

if args.eval_train or args.eval_test:   # evaluate
    if args.use_best:
        args.checkpoint_path = args.checkpoint_path + '.best'
    encoder = Encoder(num_categories=args.num_categories,
                      embed_dim=args.embed_dim,
                      p_encoder_hidden_dim=args.pencoder_hidden_dim,
                      o_encoder_hidden_dim=args.oencoder_hidden_dim)
    decoder = Decoder(num_categories=args.num_categories,
                      embed_dim=args.embed_dim,
                      p_encoder_hidden_dim=args.pencoder_hidden_dim,
                      o_encoder_hidden_dim=args.oencoder_hidden_dim,
                      p_decoder_hidden_dim=args.decoder_hidden_dim,
                      p_decoder_inner_dim=args.decoder_inner_dim)
    model = PCRNN(encoder, decoder).to(device)
    if args.eval_train:
        evaluate(model, train_loader)
    if args.eval_test:
        evaluate(model, test_loader)
