import sys
sys.path.append('.')
import datetime
import argparse
import importlib
import os
from agent import Agent
from trainer import Trainer
from config import agent_config, trainer_config
from util.buffer import ReplayBuffer
from util.setting import set_device, set_global_seed
from util.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from dynamic.transition_model import TransitionModel
import gym
import d4rl

TASK_DOMAIN = ['halfcheetah', 'hopper', 'walker2d']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="COMBO")
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
    assert task != None
    import_path = f"dynamic.static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    model_lr = trainer_config['model']['learning_rate']
    reward_penalty_coef = trainer_config['model']['reward_penalty_coef']
    model = TransitionModel(obs_space, act_space, static_fns, model_lr, reward_penalty_coef, **trainer_config)
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
                      dynamic_model = model,
                      task = args.task, 
                      **trainer_config)
    # train
    trainer.train_dynamic()
    trainer.train()
    # trainer.save_video_demo(ite = args.algo + '_' + args.task)

if __name__ == '__main__':
    train()