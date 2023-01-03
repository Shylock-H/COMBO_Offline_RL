from util.trainers import BaseTrainer
from util.setting import DEVICE
from util.buffer import ReplayBuffer
from util.helper import dict_batch_generator
import torch
import numpy as np
from tqdm import tqdm
import d4rl
from models_tf.tf_dynamics_models.fake_env import FakeEnv
from models_tf.tf_dynamics_models.constructor import format_samples_for_training

class Trainer(BaseTrainer):
    def __init__(self, 
                 agent, 
                 train_env, 
                 eval_env,
                 log,
                 offline_buffer,
                 model_buffer,
                 dynamic_model,
                 static_fns,
                 task,
                 **kwargs
                ) -> None:
        BaseTrainer.__init__(self, agent, train_env, eval_env, log, 
                             kwargs['max_train_epoch'],
                             kwargs['max_traj_len'], 
                             kwargs['eval_interval'], 
                             kwargs['log_interval'], 
                             kwargs['eval_traj_num'],
                            )                   
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self.dynamics_model = dynamic_model
        self.batch_size = kwargs['batch_size']
        self.task = task
        self.time_steps = 0

        # model
        self.model_tot_train_timesteps = 0
        self.max_model_update_epochs_to_improve = kwargs['model']['max_model_update_epochs_to_improve']
        self.model_batch_size = kwargs['model']['batch_size']
        self.hold_out_ratio = kwargs['model']['hold_out_ratio']
        self.max_model_train_iterations = kwargs['model']['max_train_iteraions']
        self.rollout_length = kwargs['model']['rollout_length']
        self.rollout_freq = kwargs['model']['rollout_freq']
        self.rollout_batch_size = kwargs['model']['rollout_batch_size']
        self.penalty_reward_coef = kwargs['model']['reward_penalty_coef']
        self.real_ratio = kwargs['model']['real_ratio']
        self.static_fns = static_fns
        self.fake_env = FakeEnv(
            self.dynamics_model,
            self.static_fns,
            penalty_coeff = self.penalty_reward_coef,
            penalty_learned_var=True
        )
        

    def train_dynamic(self):
        # get train and eval data
        max_sample_size = self.offline_buffer.allow_size
        data = self.offline_buffer.sample(max_sample_size, to_tensor = False)
        train_inputs, train_outputs = format_samples_for_training(data)
        loss = self.dynamics_model.train(
            train_inputs,
            train_outputs,
            batch_size=self.model_batch_size,
            max_epochs = self.max_model_train_iterations,
            holdout_ratio = self.hold_out_ratio
        )
        return loss
    
    def rollout(self):
        init_batch = self.offline_buffer.sample(self.rollout_batch_size, to_tensor = False)
        obs = init_batch['obs']
        for _ in range(self.rollout_length):
            action = self.agent.choose_action(obs, deterministic = False)['action']
            next_obs, reward, done, info = self.fake_env.step(obs, action)
            self.model_buffer.add_batch(obs, action, reward, next_obs, done)
            nonterm_mask = (~done).flatten()
            if nonterm_mask.sum() == 0:
                break
            obs = next_obs[nonterm_mask]

    def train(self):
        while self.trained_epochs < self.max_train_epoch:
            # rollout
            if self.trained_epochs % self.rollout_freq == 0:
                self.rollout()
            # update policy by sac
            with tqdm(total = self.max_traj_len, desc = f'Epoch : {self.trained_epochs + 1} / {self.max_train_epoch}') as t:
                while t.n < t.total:
                    real_batch_size = int(self.batch_size * self.real_ratio)
                    fake_batch_size = self.batch_size - real_batch_size
                    real_batch = self.offline_buffer.sample(real_batch_size)
                    fake_batch = self.model_buffer.sample(fake_batch_size)
                    batch = {
                        'obs' : torch.cat([real_batch['obs'], fake_batch['obs']], dim = 0),
                        'action' : torch.cat([real_batch['action'], fake_batch['action']], dim = 0),
                        'reward' : torch.cat([real_batch['reward'], fake_batch['reward']], dim = 0),
                        'next_obs' : torch.cat([real_batch['next_obs'], fake_batch['next_obs']], dim = 0),
                        'done' : torch.cat([real_batch['done'], fake_batch['done']], dim = 0)
                    }
                    loss = self.agent.update(batch)
                    t.set_postfix(**loss)
                    t.update(1)
            # eval and log
            if self.trained_epochs > 0  and self.trained_epochs % self.eval_interval == 0:
                dict = self.eval()
                for key, item in dict.items():
                    self.log.record(key, item, self.trained_epochs)
                    if key == 'performance/eval_return':
                        self.log.record(f'normalize_score/{self.task}', d4rl.get_normalized_score(self.task, item), self.trained_epochs)
            
            if self.trained_epochs > 0 and self.trained_epochs % self.log_interval == 0:
                for key, item in loss.items():
                    self.log.record(key, item, self.trained_epochs)

            self.trained_epochs += 1