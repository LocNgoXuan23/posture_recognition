from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional
from torch.autograd import Variable

def reduce_loss(loss, reduction='mean'):
	return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
	def __init__(self, ε:float=0.1, reduction='mean'):
		super().__init__()
		self.ε,self.reduction = ε,reduction
	
	def forward(self, output, target):
		# number of classes
		c = output.size()[-1]
		log_preds = F.log_softmax(output, dim=-1)
		loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
		nll = F.nll_loss(log_preds, target, reduction=self.reduction)
		# (1-ε)* H(q,p) + ε*H(u,p)
		# print((1-self.ε)*nll + self.ε*(loss/c))
		return (1-self.ε)*nll + self.ε*(loss/c) 

class SoftTargetCrossEntropy(nn.Module):

	def __init__(self):
		super(SoftTargetCrossEntropy, self).__init__()

	def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

		loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
		return loss.mean()

def reduce_loss(loss, reduction='mean'):
	return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class BinaryCrossEntropy(nn.Module):
	""" BCE with optional one-hot from dense targets, label smoothing, thresholding
	NOTE for experiments comparing CE to BCE /w label smoothing, may remove
	"""
	def __init__(
			self, smoothing=0.1, target_threshold: Optional[float] = None, weight: Optional[torch.Tensor] = None,
			reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
		super(BinaryCrossEntropy, self).__init__()
		assert 0. <= smoothing < 1.0
		self.smoothing = smoothing
		self.target_threshold = target_threshold
		self.reduction = reduction
		self.register_buffer('weight', weight)
		self.register_buffer('pos_weight', pos_weight)

	def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		assert x.shape[0] == target.shape[0]
		if target.shape != x.shape:
			# NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
			num_classes = x.shape[-1]
			# FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
			off_value = self.smoothing / num_classes
			on_value = 1. - self.smoothing + off_value
			target = target.long().view(-1, 1)
			target = torch.full(
				(target.size()[0], num_classes),
				off_value,
				device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
		if self.target_threshold is not None:
			# Make target 0, or 1 if threshold set
			target = target.gt(self.target_threshold).to(dtype=target.dtype)
		return F.binary_cross_entropy_with_logits(
			x, target,
			self.weight,
			pos_weight=self.pos_weight,
			reduction=self.reduction)

class FocalLoss(nn.Module):
	def __init__(self, gamma=0, alpha=None, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
		if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)

		logpt = F.log_softmax(input)
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type()!=input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0,target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1-pt)**self.gamma * logpt
		if self.size_average: return loss.mean()
		else: return loss.sum()

class AMSoftmax(nn.Module):
	def __init__(self,
				 in_feats=1000,
				 n_classes=23,
				 m=0.3,
				 s=15):
		super(AMSoftmax, self).__init__()
		self.m = m
		self.s = s
		self.in_feats = in_feats
		self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
		self.ce = nn.CrossEntropyLoss()
		nn.init.xavier_normal_(self.W, gain=1)

	def forward(self, x, lb):
		assert x.size()[0] == lb.size()[0]
		assert x.size()[1] == self.in_feats
		x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
		x_norm = torch.div(x, x_norm)
		w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
		w_norm = torch.div(self.W, w_norm)
		costh = torch.mm(x_norm, w_norm)
		lb_view = lb.view(-1, 1)
		if lb_view.is_cuda: lb_view = lb_view.cpu()
		delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
		if x.is_cuda: delt_costh = delt_costh.cuda()
		costh_m = costh - delt_costh
		costh_m_s = self.s * costh_m
		loss = self.ce(costh_m_s, lb)
		return loss

class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""
	def __init__(self, temperature=0.07, contrast_mode='all',
				 base_temperature=0.07):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature

	def forward(self, features, labels=None, mask=None):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf
		Args:
			features: hidden vector of shape [bsz, n_views, ...].
			labels: ground truth of shape [bsz].
			mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
				has the same class as sample i. Can be asymmetric.
		Returns:
			A loss scalar.
		"""
		device = (torch.device('cuda')
				  if features.is_cuda
				  else torch.device('cpu'))

		if len(features.shape) < 3:
			raise ValueError('`features` needs to be [bsz, n_views, ...],'
							 'at least 3 dimensions are required')
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			mask = mask.float().to(device)

		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
		if self.contrast_mode == 'one':
			anchor_feature = features[:, 0]
			anchor_count = 1
		elif self.contrast_mode == 'all':
			anchor_feature = contrast_feature
			anchor_count = contrast_count
		else:
			raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T),
			self.temperature)
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count)
		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
			0
		)
		mask = mask * logits_mask

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

		# loss
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()

		return loss

class ContrastiveLoss(Module):
	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
		loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
									  label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive