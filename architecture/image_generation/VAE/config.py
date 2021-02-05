import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'easyvqa'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.IM_SIZE = 64

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 601
__C.TRAIN.VAE_LR = 0.0002
__C.TRAIN.G_LR = 0.0002
__C.TRAIN.D_LR = 0.0002
__C.TRAIN.CHECKPOINT = ''
__C.TRAIN.VAE_CHECKPOINT = ''

__C.MODEL = edict()
__C.MODEL.NF = 8
__C.MODEL.Z_DIM = 100
__C.MODEL.EF_DIM = 10
__C.MODEL.EF_TYPE = "sbert"
__C.MODEL.GAN = "DFGAN"
__C.TEST = edict()
__C.TEST.CHECKPOINT = ''


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
