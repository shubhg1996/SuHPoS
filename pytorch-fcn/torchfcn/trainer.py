import datetime
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def pose_loss(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    sft_p = F.softmax(input)
    mask = target >= 0
    target = target[mask]
    sft_p = sft_p[mask]
    loss = F.mse_loss(sft_p, target, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def pose_paf_metric(input_pose, input_paf,mat,image)
    n, c, h, w = input_pose.size()
    # log_p: (n, c, h, w)
    sft_p = F.softmax(input_pose)
    #mask = target >= 0
    #target = target[mask]
    #sft_p = sft_p[mask]
    for part in range(c):
	x_list = []
	y_list = []
	map_ori = stft_p[:,:,part]
	map = gaussian_filter(map_ori, sigma=3)

	map_left = np.zeros(map.shape)
	map_left[1:,:] = map[:-1,:]
	map_right = np.zeros(map.shape)
	map_right[:-1,:] = map[1:,:]
	map_up = np.zeros(map.shape)
	map_up[:,1:] = map[:,:-1]
	map_down = np.zeros(map.shape)
	map_down[:,:-1] = map[:,1:]

	peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > 0.1))
	peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
	peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
	id = range(peak_counter, peak_counter + len(peaks))
	peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

	all_peaks.append(peaks_with_score_and_id)
	peak_counter += len(peaks)

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq =  [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]
    # the middle joints heatmap correpondence
    
    connection_all = []
    special_k = []
    mid_num = 10
    for k in range(len(limbSeq)):
	score_midx = input_paf[limbSeq[k][0],:,:]
	score_midy = input_paf[limbSeq[k][1],:,:]
	candA = all_peaks[limbSeq[k][0]]
	candB = all_peaks[limbSeq[k][1]]
	nA = len(candA)
	nB = len(candB)
	indexA, indexB = limbSeq[k]
	if(nA != 0 and nB != 0):
	connection_candidate = []
	for i in range(nA):
	    for j in range(nB):
		vec = np.subtract(candB[j][:2], candA[i][:2])
		norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
		vec = np.divide(vec, norm)

		startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
			       np.linspace(candA[i][1], candB[j][1], num=mid_num))

		vec_x = np.array([score_midx[int(round(startend[I][1])), int(round(startend[I][0]))] \
				  for I in range(len(startend))])
		vec_y = np.array([score_midy[int(round(startend[I][1])), int(round(startend[I][0]))] \
				  for I in range(len(startend))])

		score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
		score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
		criterion1 = len(np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(score_midpts)
		criterion2 = score_with_dist_prior > 0
		if criterion1 and criterion2:
		    connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

	connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
	connection = np.zeros((0,5))
	for c in range(len(connection_candidate)):
	    i,j,s = connection_candidate[c][0:3]
	    if(i not in connection[:,3] and j not in connection[:,4]):
		connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
		if(len(connection) >= min(nA, nB)):
		    break

	connection_all.append(connection)
	else:
	special_k.append(k)
	connection_all.append([])	
	
    subset = -1 * np.ones((0, 15))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(limbSeq)):
	if k not in special_k:
	partAs = connection_all[k][:,0]
	partBs = connection_all[k][:,1]
	indexA, indexB = np.array(limbSeq[k]) - 1

	for i in range(len(connection_all[k])): #= 1:size(temp,1)
	    found = 0
	    subset_idx = [-1, -1]
	    for j in range(len(subset)): #1:size(subset,1):
		if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
		    subset_idx[found] = j
		    found += 1

	    if found == 1:
		j = subset_idx[0]
		if(subset[j][indexB] != partBs[i]):
		    subset[j][indexB] = partBs[i]
		    subset[j][-1] += 1
		    subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
	    elif found == 2: # if found 2 and disjoint, merge them
		j1, j2 = subset_idx
		print "found = 2"
		membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
		if len(np.nonzero(membership == 2)[0]) == 0: #merge
		    subset[j1][:-2] += (subset[j2][:-2] + 1)
		    subset[j1][-2:] += subset[j2][-2:]
		    subset[j1][-2] += connection_all[k][i][2]
		    subset = np.delete(subset, j2, 0)
		else: # as like found == 1
		    subset[j1][indexB] = partBs[i]
		    subset[j1][-1] += 1
		    subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

	    # if find no partA in the subset, create a new subset
	    elif not found and k < 17:
		row = -1 * np.ones(15)
		row[indexA] = partAs[i]
		row[indexB] = partBs[i]
		row[-1] = 2
		row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
		subset = np.vstack([subset, row])	
    
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    H,W,_ = image.shape
    Hn = np.ceil(H/32).astype(np.int64)
    Wn = np.ceil(W/32).astype(np.int64)
    pck = 0
    count = 1
    for n in range(len(subset)):
	indexes = subset[n][:-2]
	min_dist = float("inf")
	x_pred = candidate[indexes.astype(int),1]
	y_pred = candidate[indexes.astype(int),0]
	for human in mat['joints'][0]:
	    poselist = np.around(human[:,:-1]).astype(np.int64)
            poselist[:,0] = poselist[:,0]*Hn/H
            poselist[:,1] = poselist[:,1]*Wn/W
	    dist = ((x_pred - poselist[:,0])**2 + (y_pred - poselist[:,1])**2)**0.5
	    if dist<min_dist:
		min_dist = dist
		closest_human = poselist
		xd = np.square([x - poselist[i - 1,0] for i, x in enumerate(poselist[:,0]) if i > 0])
		yd = np.square([y - poselist[i - 1,1] for i, y in enumerate(poselist[:,1]) if i > 0])
		d = [sum(x) for x in zip(xd, yd)]
		PCK_t = 2*min(d)**0.5
		
	for i in range(13):
		index = indexes[i]
		x_p = candidate[index.astype(int),1]
		y_p = candidate[index.astype(int),0]
		x_gt = poselist[i,0]
		y_gt = poselist[i,1]
		dist == ((x_p - x_gt)**2 + (y_p - y_gt)**2)**0.5
		if dist < PCK_t:
			pck = (pck*count + dist)/(count + 1)
			count = count + 1
    return pck		
	    	

def paf_loss(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    mask = target >= 0
    target = target[mask]
    sft_p = input[mask]
    loss = F.mse_loss(sft_p, target, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

	    #print(score.size(),target.size(),data.size())
            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)
	    #print(score.size(),target.size(),data.size())
            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss /= len(data)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    torchfcn.utils.label_accuracy_score(
                        [lt], [lp], n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

                
class TrainerPAF(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'valid/loss',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.pose_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, targetpose, targetpaf) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, targetpose, targetpaf = data.cuda(), targetpose.cuda(), targetpaf.cuda()
            data, targetpose, targetpaf = Variable(data, volatile=True), Variable(targetpose), Variable(targetpaf)
            scorepose, scorepaf = self.model(data)

            losspose = pose_loss(scorepose, targetpose,
                                   size_average=self.size_average)
            losspaf = paf_loss(scorepaf, targetpaf,
                                   size_average=self.size_average)
            loss = losspose+losspaf
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)

            val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + [''] + \
                  [val_loss] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = 0
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.pose_names)

        for batch_idx, (data, targetpose, targetpaf) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            if self.cuda:
                data, targetpose, targetpaf = data.cuda(), targetpose.cuda(), targetpaf.cuda()
            data, targetpose, targetpaf = Variable(data, volatile=True), Variable(targetpose), Variable(targetpaf)
            self.optim.zero_grad()
            scorepose, scorepaf = self.model(data)

            losspose = pose_loss(scorepose, targetpose,
                                   size_average=self.size_average)
            losspaf = paf_loss(scorepaf, targetpaf,
                                   size_average=self.size_average)
            loss = losspose+losspaf
            loss /= len(data)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + \
                    [''] + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
