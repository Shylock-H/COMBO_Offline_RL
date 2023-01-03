from util.trainers import BaseTrainer
from util.setting import DEVICE
from util.buffer import ReplayBuffer
from util.helper import dict_batch_generator
import torch
import numpy as np
from tqdm import tqdm
import d4rl

class Trainer(BaseTrainer):
    def __init__(self, 
                 agent, 
                 train_env, 
                 eval_env,
                 log,
                 offline_buffer,
                 model_buffer,
                 dynamic_model,
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
        if self.max_model_train_iterations == None:
            self.max_model_train_iterations = np.inf
        self.rollout_length = kwargs['model']['rollout_length']
        self.rollout_freq = kwargs['model']['rollout_freq']
        self.rollout_batch_size = kwargs['model']['rollout_batch_size']
        self.real_ratio = kwargs['model']['real_ratio']
        

    def train_dynamic(self):
        # get train and eval data
        max_sample_size = self.offline_buffer.allow_size
        num_train_data = int(max_sample_size * (1.0 - self.hold_out_ratio))
        env_data = self.offline_buffer.sample(batch_size = max_sample_size)
        train_data, eval_data = {}, {}
        for key in env_data.keys():
            train_data[key] = torch.tensor(env_data[key][:num_train_data]).to(DEVICE)
            eval_data[key] = torch.tensor(env_data[key][num_train_data:]).to(DEVICE)
        self.dynamics_model.reset_normalizers()
        self.dynamics_model.update_normalizer(train_data['obs'], train_data['action'])

        # train model
        model_train_iters = 0
        model_train_epochs = 0
        num_epochs_since_prev_best = 0
        break_training = False
        self.dynamics_model.reset_best_snapshots()

        # init eval_mse_losses
        self.log.print("Start training dynamics")
        eval_mse_losses, _ = self.dynamics_model.eval_data(eval_data, update_elite_models=False)
        self.log.record("loss/model_eval_mse_loss", eval_mse_losses.mean(), self.model_tot_train_timesteps)
        updated = self.dynamics_model.update_best_snapshots(eval_mse_losses)
        while not break_training:
    
            for train_data_batch in dict_batch_generator(train_data, self.model_batch_size):
                
                model_log_infos = self.dynamics_model.update(train_data_batch)
                model_train_iters += 1
                self.model_tot_train_timesteps += 1
                
            eval_mse_losses, _ = self.dynamics_model.eval_data(eval_data, update_elite_models=False)
            self.log.record("loss/model_eval_mse_loss", eval_mse_losses.mean(), self.model_tot_train_timesteps)
            updated = self.dynamics_model.update_best_snapshots(eval_mse_losses)
            num_epochs_since_prev_best += 1

            if updated:
                model_train_epochs += num_epochs_since_prev_best
                num_epochs_since_prev_best = 0
            if num_epochs_since_prev_best >= self.max_model_update_epochs_to_improve or model_train_iters > self.max_model_train_iterations\
                    or self.model_tot_train_timesteps > 1000000:
                break
            # Debug
        self.dynamics_model.load_best_snapshots()
       
        # evaluate data to update the elite models
        self.dynamics_model.eval_data(eval_data, update_elite_models=True)
        model_log_infos['misc/norm_obs_mean'] = torch.mean(torch.Tensor(self.dynamics_model.obs_normalizer.mean)).item()
        model_log_infos['misc/norm_obs_var'] = torch.mean(torch.Tensor(self.dynamics_model.obs_normalizer.var)).item()
        model_log_infos['misc/norm_act_mean'] = torch.mean(torch.Tensor(self.dynamics_model.act_normalizer.mean)).item()
        model_log_infos['misc/norm_act_var'] = torch.mean(torch.Tensor(self.dynamics_model.act_normalizer.var)).item()
        model_log_infos['misc/model_train_epochs'] = model_train_epochs
        model_log_infos['misc/model_train_train_steps'] = model_train_iters
        
        return model_log_infos
    
    def rollout(self):
        init_batch = self.offline_buffer.sample(self.rollout_batch_size)
        obs = init_batch['obs']
        for _ in range(self.rollout_length):
            action = self.agent.choose_action(obs, deterministic = False)['action']
            next_obs, reward, done, info = self.dynamics_model.predict(obs, action)
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

        # tj_returns = []
        # tj_lengths = []

        # while self.trained_epochs < self.max_train_epoch:
        #     tj_return = 0
        #     tj_length = 0
        #     state = self.train_env.reset()
            
        #     for _ in range(self.max_traj_len):
                
        #         if self.trained_epochs < self.random_epoch:
        #             action = self.train_env.action_space.sample()
        #         else:
        #             action = self.agent.choose_action(state)['action']
        #         next_state, reward, done, info = self.train_env.step(action)
        #         self.buffer.add_transition(state, action, reward, next_state, done)
        #         tj_return += reward
        #         tj_length += 1
        #         # update
        #         if self.buffer.allow_size > 5 * self.batch_size:
        #             batch = self.buffer.sample(self.batch_size)
        #             self.agent.update(batch)
                
        #         self.time_steps += 1

        #         if done:
        #             break
        #         else:
        #             state = next_state

        #     tj_lengths.append(tj_length)
        #     tj_returns.append(tj_return)

        #     if self.trained_epochs > 0  and self.trained_epochs % self.eval_interval == 0:
        #         dict = self.eval()
        #         for key, item in dict.items():
        #             self.log.record(key, item, self.trained_epochs)
        #     # log 
        #     if self.trained_epochs > 0 and self.trained_epochs % self.log_interval == 0:
        #         self.log.record('train/returns', tj_return, self.trained_epochs)
        #         self.log.record('train/lengths', tj_length, self.trained_epochs)
                
        #     self.trained_epochs += 1
        
        # # plt.plot(tj_returns)
        # # plt.savefig('tmp.png')

        # return {
        #     'train/lengths' : tj_lengths,
        #     'train/returns' : tj_returns
        # }