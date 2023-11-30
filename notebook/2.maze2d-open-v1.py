import os
import sys

import gym
import d4rl
from PIL import Image
import numpy as np

horizon = 128

env_name = 'maze2d-open-v1'
device = 1
savepath = f"logs/maze2d-open-v2/wo_conditon_H{horizon}"
os.makedirs(savepath, exist_ok=True)

# get dataset
from diffuser.datasets import GoalDataset, UnconditionedDataset

dataset = UnconditionedDataset(env=env_name, termination_penalty=None, 
                      horizon=horizon,
                      normalizer='LimitsNormalizer', 
                      preprocess_fns=[],
                      use_padding=False,
                      max_path_length=40000
                      )

# Get renderer
from diffuser.utils.rendering import Maze2dRenderer

renderer = Maze2dRenderer(env_name)

# Get model
from diffuser.models import TemporalUnet

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

model = TemporalUnet(horizon=horizon, 
                     transition_dim=observation_dim + action_dim,
                     cond_dim=observation_dim,
                     dim_mults=(1, 4, 8),
                     )
model.to(device=device);

# Get diffusion model
from diffuser.models import GaussianDiffusion

diffusion = GaussianDiffusion(
    model=model,
    horizon=horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=64,
    loss_type='l2',
    clip_denoised=True,
    predict_epsilon=False,
    action_weight=1,
    loss_weights=None,
    loss_discount=1,
)
diffusion.to(device=device);

# Get Trainer
from diffuser.utils import Trainer

trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset,
    renderer=renderer,
    train_batch_size=64,
    train_lr=2e-4,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    sample_freq=1000,
    save_freq=1000,
    label_freq=int(2e6//50),
    save_parallel=False,
    results_folder=savepath,
    bucket=None,
    n_reference=50,
    n_samples=10,
)

from diffuser import utils

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0], device=device)
print(batch.trajectories.device)
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

# now train
n_epochs = int(2e5 // 10000)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {savepath}')
    trainer.train(n_train_steps=10000)