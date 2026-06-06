import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD='foolsgold'
AGGR_FLTRUST    ='fltrust'
AGGR_FLDETECTOR = 'fldetector'
AGGR_LEADFL = 'leadfl'
AGGR_FLIP = 'flip'
MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20

# Defense hyper-parameters (paper defaults)
FLDETECTOR_DEFAULTS = {
	'fldetector_mad_k': 2.5,
}

LEADFL_DEFAULTS = {
	'leadfl_gamma': 1.0,
	'leadfl_keep_ratio': 0.7,
	'leadfl_weight_floor': 1e-4,
}

FLIP_DEFAULTS = {
	'flip_cos_threshold': -0.05,
	'flip_max_norm_ratio': 2.0,
}

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'