import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import data
import models
from data import TrainSplit, EvaluateSplit
from args import parse_args


args = parse_args()

# Ensure that the kernel for RNN is greated than the number of proposals
assert (args.W > args.K)

# Check if directory exists and create one if it doesn't:
if not os.path.isdir(args.save):
    os.makedirs(args.save)

# Argument hack
args.shuffle = args.shuffle != 0

# Clean the directory
if args.clean:
    for f in ['model.pth', 'train.log', 'val.log', 'test.log']:
        try:
            os.remove(os.path.join(args.save, f))
        except:
            continue

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
elif args.cuda:
    raise Exception("No GPU found, please run without --cuda")

# Save the arguments for future viewing
with open(os.path.join(args.save, 'args.json'), 'w') as f:
    f.write(json.dumps(vars(args)))

###############################################################################
# Load data
###############################################################################

print("| Loading data into corpus: %s" % args.data)
dataset = getattr(data, args.dataset)(args)
w1 = dataset.pos_prop_weight
train_dataset = TrainSplit(dataset.training_ids, dataset, args)
val_dataset = EvaluateSplit(dataset.validation_ids, dataset, args)
train_val_dataset = EvaluateSplit(dataset.training_ids, dataset, args)

print("| Dataset created")
train_loader = DataLoader(train_dataset, shuffle=args.shuffle, batch_size=args.batch_size, num_workers=args.nthreads,
                          collate_fn=train_dataset.collate_fn)
train_evaluator = DataLoader(train_val_dataset, shuffle=args.shuffle, batch_size=1, num_workers=args.nthreads,
                             collate_fn=val_dataset.collate_fn)
val_evaluator = DataLoader(val_dataset, shuffle=args.shuffle, batch_size=1, num_workers=args.nthreads,
                           collate_fn=val_dataset.collate_fn)
print("| Data Loaded: # training data: %d, # val data: %d" % (len(train_loader) * args.batch_size, len(val_evaluator)))

###############################################################################
# Build the model
###############################################################################

if args.resume:
    model = torch.load(os.path.join(args.save, 'model.pth'))
else:
    model = models.SST(
            video_dim=args.video_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            W=args.W,
            K=args.K,
            rnn_type=args.rnn_type,
            rnn_num_layers=args.rnn_num_layers,
            rnn_dropout=args.rnn_dropout,
    )

if args.cuda:
    model.cuda()


###############################################################################
# Training code
###############################################################################


def iou(interval, featstamps, return_index=False):
    start_i, end_i = interval[0], interval[1]
    output = 0.0
    gt_index = -1
    for i, (start, end) in enumerate(featstamps):
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
        overlap = float(intersection) / (union + 1e-8)
        if overlap >= output:
            output = overlap
            gt_index = i
    if return_index:
        return output, gt_index
    return output


def proposals_to_timestamps(proposals, duration, num_proposals):
    # if num_proposals is None, extract all possible proposals
    _, nb_steps, K = proposals.size()
    if num_proposals and num_proposals < nb_steps * K:
        # keep only top num_proposals proposals
        sort, _ = proposals.view(nb_steps * K).sort()
        score_threshold = sort[-num_proposals]
        proposals = proposals >= score_threshold
    step_length = duration / nb_steps
    timestamps = []
    for time_step in np.arange(nb_steps):
        p = proposals[0, time_step]
        if p.sum() != 0:  # ie we have non zero score at this step
            end = time_step * step_length
            for k in np.arange(K):
                if p[k] != 0:
                    start = max(0, time_step - k - 1) * step_length
                    timestamps.append((start, end))
    return timestamps


def calculate_stats(proposals, gt_times, duration, args):
    timestamps = proposals_to_timestamps(proposals.data, duration, args.num_proposals)
    gt_detected = np.zeros(len(gt_times))
    for i, timestamp in enumerate(timestamps):
        iou_i, k = iou(timestamp, gt_times, return_index=True)
        if iou_i > args.iou_threshold:
            gt_detected[k] = 1
    return gt_detected.sum()*1./len(gt_detected)


def evaluate(data_loader, maximum=None):
    model.eval()
    total = len(data_loader)
    if maximum is not None:
        total = min(total, maximum)
    recall = np.zeros(total)
    for batch_idx, (features, gt_times, duration) in enumerate(data_loader):
        if maximum is not None and batch_idx >= maximum:
            break
        if args.cuda:
            features = features.cuda()
        features = Variable(features)
        proposals = model(features)
    recall[batch_idx] = calculate_stats(proposals, gt_times, duration, args)
    return np.mean(recall)


def train(epoch, w1):
    model.train()
    total_loss = []
    model.train()
    start_time = time.time()
    for batch_idx, (features, masks, labels) in enumerate(train_loader):
        if args.cuda:
            features = features.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
        features = Variable(features)
        # print(repr(features))
        # raise ValueError()
        optimizer.zero_grad()
        proposals = model(features)
        loss = model.compute_loss_with_BCE(proposals, masks, labels, w1)
        loss.backward()
        optimizer.step()
        # ratio of weights updates to debug 
        # for group in optimizer.param_groups:
        # for p in group['params']:
        # print("ratio of weights update ")
        # print(p.grad.div(p).mean().data)
        total_loss.append(loss.data[0])

        # Debugging training samples
        if args.debug:
            recall = evaluate(train_evaluator, maximum=args.num_vids_eval)
            log_entry = ('| train recall@{}-iou={}: {:2.4f}\%'.format(args.num_proposals, args.iou_threshold, recall))
            print(log_entry)

        # Print out training loss every interval in the batch
        if batch_idx % args.log_interval == 0:  # and batch_idx > 0:
            cur_loss = total_loss[-1]
            elapsed = time.time() - start_time
            log_entry = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | ' \
                        'loss {:5.6f}'.format(
                    epoch, batch_idx, len(train_loader), args.lr,
                    elapsed * 1000 / args.log_interval, cur_loss * 1000)
            print(log_entry)
            with open(os.path.join(args.save, 'train.log'), 'a') as f:
                f.write(log_entry)
                f.write('\n')
            start_time = time.time()


# Loop over epochs.
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(epoch, w1)
    recall = evaluate(val_evaluator, maximum=args.num_vids_eval)
    print('-' * 89)
    log_entry = ('| end of epoch {:3d} | time: {:5.2f}s | val recall@{}-iou={}: {:2.2f}\%'.format(
            epoch, (time.time() - epoch_start_time), args.num_proposals, args.iou_threshold, recall))
    print(log_entry)
    print('-' * 89)
    with open(os.path.join(args.save, 'val.log'), 'a') as f:
        f.write(log_entry)
        f.write('\n')
    if args.save != '' and epoch % args.save_every == 0 and epoch > 0:
        torch.save(model, os.path.join(args.save, 'model_' + str(epoch) + '.pth'))

# Run on test data and save the model.
# This is not needed now since test videos have no proposals 
# print("| Testing model on test set")
# test_dataset = EvaluateSplit(dataset.testing_ids, dataset, args)
# test_evaluator = DataLoader(test_dataset, shuffle=args.shuffle, batch_size=1, num_workers=args.nthreads, collate_fn=test_dataset.collate_fn)
# test_precision, test_recall = evaluate(test_evaluator)
# print('=' * 89)
# print('| End of training | test precision {:2.2f}\% | test recall {:2.2f}\%'.format(
#    test_precision, test_recall))
# print('=' * 89)
# if args.save != '':
#    torch.save(model, os.path.join(args.save, 'model.pth'))
