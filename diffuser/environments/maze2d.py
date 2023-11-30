import os
import numpy as np
from d4rl.pointmaze.maze_model import MazeEnv, OPEN
import collections
from tqdm import tqdm
import h5py

def insert_transitions(env, start_loc_org, target_loc_org, time_step, p_gain, d_gain, dataset, num_episode=5000):
    for _ in tqdm(range(num_episode)):
        start_loc = start_loc_org + np.random.uniform(low=-.1, high=.1, size=start_loc_org.shape)
        target_loc = target_loc_org + np.random.uniform(low=-.1, high=.1, size=target_loc_org.shape)
        
        ratio = np.sort(np.random.uniform(size=time_step))
        ref_trajectory = np.stack([np.interp(ratio, [0, 1], [start_loc[0], target_loc[0]]), np.interp(ratio, [0, 1], [start_loc[1], target_loc[1]])], axis=1)
        ref_trajectory = ref_trajectory + 0.2 * np.random.randn(*ref_trajectory.shape)
        ref_trajectory = np.concatenate([ref_trajectory, np.expand_dims(target_loc, axis=0).repeat([env.max_episode_steps - time_step], axis=0)])
        
        env.reset()
        env.set_target(target_loc)
        state = env.reset_to_location(start_loc)
        obs = [state]
        for i, ref_pos in enumerate(ref_trajectory):
            pos, vel = state[0:2], state[2:4]
            
            prop = ref_pos - pos
            action = p_gain * prop + d_gain * vel
            next_state, reward, done, info = env.step(action)
            obs.append(next_state)
            
            dataset['actions'].append(action)
            dataset['observations'].append(state)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(reward > 0)
            dataset['timeouts'].append(i >= env.max_episode_steps-1 and not reward > 0)
            dataset['info/goal'].append(env.get_target())
            dataset['info/qpos'].append(state[:2])
            dataset['info/qvel'].append(state[2:4])
            
            state = next_state
            
            if reward > 0 or i >= env.max_episode_steps-1:
                break

class Maze2dOpenEnv(MazeEnv):

    def __init__(self):
        MazeEnv.__init__(self, maze_spec=OPEN, reward_type='sparse', reset_target=False, ref_min_score=0.01, ref_max_score=20.66, dataset_url='http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5')
        
    def get_dataset(self, h5path=None):
        
        dataname = "maze2d-open-sparse-v2"
        h5path = os.path.join("/home/cspark/.d4rl/datasets/", f"{dataname}.hdf5")
        
        if os.path.exists(h5path):    
            dataset = super().get_dataset(h5path=h5path)
        else:
        
            wrapped_env = Maze2dOpenEnv()
            env = wrapped_env
            env.max_episode_steps = 150
            
            # control 
            time_step = 140
            p_gain = 10.0
            d_gain = -1.0

            dataset = collections.defaultdict(list)
            
            print(f"getting dataset for {dataname}")
            start_loc_org = np.array([1, 1], dtype=float)
            target_loc_org = np.array([3, 5], dtype=float)
            insert_transitions(env, start_loc_org, target_loc_org, time_step, p_gain, d_gain, dataset)
            
            start_loc_org = np.array([3, 1], dtype=float)
            target_loc_org = np.array([1, 5], dtype=float)
            insert_transitions(env, start_loc_org, target_loc_org, time_step, p_gain, d_gain, dataset)
            

            for key, val in dataset.items():
                dataset[key] = np.stack(val, axis=0)
            
            save_dataset_h5py = h5py.File(h5path, 'w')
            for k in dataset:
                save_dataset_h5py.create_dataset(k, data=dataset[k], compression='gzip')
            
        return dataset