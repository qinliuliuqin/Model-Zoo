from easydict import EasyDict as edict
from segmentation3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer


__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-segmentation pair list
__C.general.imseg_list = '/home/admin/test/T4.0/Medical-Segmentation3d-Toolkit/train_teeth.txt' #'/home/admin/test/TestSeg-V3.0/Medical-Segmentation3d-Toolkit/datasets/train.txt' # 

# the output of training models and logs
__C.general.save_dir = '/home/admin/test/T4.0/Medical-Segmentation3d-Toolkit/models/model_0803_dental'

# the model scale
__C.general.model_scale = 'fine'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = 550

# the number of GPUs used in training. Set to 0 if using cpu only.
__C.general.num_gpus = 3

# random seed used in training (debugging purpose)
__C.general.seed = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 3

# the resolution on which segmentation is performed
__C.dataset.spacing =[0.4, 0.4, 0.4] #[[2.0, 2.0, 2.0] #0.8, 0.8, 0.8] # 

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [128, 128, 128] #[80, 80, 80] #[96, 96, 96] #

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
# 2) MASK: sampling crops randomly within segmentation mask
# 3) HYBRID: Sampling crops randomly with both GLOBAL and MASK methods
# 4) CENTER: sampling crops in the image center
__C.dataset.sampling_method = 'GLOBAL'

# translation augmentation (unit: mm)
__C.dataset.random_translation = [10, 10, 10] #[15, 15, 5] # [64, 64, 5] #

# spacing scale augmentation, spacing scale will be randomly selected from [min, max]
# during training, the image spacing will be spacing * scale
__C.dataset.random_scale = [0.8, 1.2]

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [AdaptiveNormalizer()]

##################################
# training loss
##################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity Loss which supports binary and multi-class segmentation
__C.loss.name = 'Focal'

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1/4, 3/8, 3/8] # [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]#

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2


##################################
# net
##################################

__C.net = {}

# the network name
__C.net.name = 'vbnet'

# enable uncertainty by trun on drop out layers in the segmentation net
__C.net.dropout_turn_on = False

##################################
# training parameters
##################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 3001

# the number of samples in a batch
__C.train.batchsize = 6

# the number of threads for IO
__C.train.num_threads = 6

# the learning rate
__C.train.lr = 1e-4

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

# the number of batches to save model
__C.train.save_epochs = 50


###################################
# debug parameters
###################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = False
