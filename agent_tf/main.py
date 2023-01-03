import sys
sys.path.append('.')
import datetime
import argparse
import importlib
import os
from agent import Agent
from agent_tf.trainer import Trainer
from config import agent_config, trainer_config
from util.buffer import ReplayBuffer
from util.setting import set_device, set_global_seed
from util.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from models_tf.tf_dynamics_models.constructor import construct_model
import gym
import d4rl
import numpy as np

TASK_DOMAIN = ['halfcheetah', 'hopper', 'walker2d']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="COMBO-tf")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type = str, default = 'log')

    return parser.parse_args()

def train(args = get_args()):
    set_device()
    set_global_seed(args.seed)
    # env
    env = gym.make(args.task)
    env.seed(seed = args.seed)
    dataset = d4rl.qlearning_dataset(env)

    obs_space = env.observation_space
    act_space = env.action_space
    # buffer
    offline_buffer = ReplayBuffer(obs_space, act_space, buffer_size = len(dataset['observations']))
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(obs_space, act_space, buffer_size = len(dataset['observations']))
    # agent
    agent = Agent(obs_space, act_space, **agent_config)
    # dynamic model
    task = None
    for key in args.task.split('-'):
        if key.lower() in TASK_DOMAIN:
            task = key.lower()
            break
    import_path = f"dynamic.static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    dynamic_model = construct_model(
        obs_dim = np.prod(env.observation_space.shape),
        act_dim = np.prod(env.action_space.shape),
        hidden_dim = 200,
        separate_mean_var = True
    )
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo}'
    log_path = os.path.join(args.logdir, args.task, args.algo, log_file)
    writer = SummaryWriter(log_path)
    logger = Logger(writer)
    # trainer
    trainer = Trainer(agent,
                      train_env = env, 
                      eval_env = env,
                      log = logger,
                      offline_buffer = offline_buffer,
                      model_buffer = model_buffer,
                      dynamic_model = dynamic_model,
                      static_fns = static_fns,
                      task = args.task, 
                      **trainer_config)
    # train
    trainer.train_dynamic()
    trainer.train()
    # trainer.save_video_demo(ite = args.algo + '_' + args.task)

if __name__ == '__main__':
    train()