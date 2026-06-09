from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_s_gamma_05"
config.resume = False
config.output = 'synthetic_baseline_run/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.weight_decay = 0.05
config.batch_size = 64
config.optimizer = "adamw"
config.lr = 6e-3
config.verbose = 10
config.dali = False # DALI must be false for synthetic
config.dali_aug = False

config.num_workers = 2

# Use the built-in synthetic dataset
config.rec = "synthetic"
config.num_classes = 100
config.num_image = 2000
config.num_epoch = 2
config.warmup_epoch = 0
config.val_targets = []
config.frequent = 2
config.verbose = 5
