from submodule.controlnet.share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scripts.data.dataset import BaseShapeDataset
from submodule.controlnet.cldm.logger import ImageLogger
from submodule.controlnet.cldm.model import create_model, load_state_dict


def test_dataset(dataset):
    item = dataset[500]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)

# Configs
resume_path = 'checkpoints/diffusion/control_sd21_ini.ckpt'
batch_size = 8
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

dataset = BaseShapeDataset('dataset/LeafData',1000)
# test_dataset(dataset)
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('submodule/controlnet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=[3], precision=32, weights_save_path='./results/controlnet',callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
