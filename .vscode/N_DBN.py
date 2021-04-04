'''
Author: sunkaiwei
Date: 2021-04-04 16:49:40
LastEditTime: 2021-04-04 17:40:09
LastEditer: sunkaiwei
FilePath: \DBN\.vscode\N_DBN.py
Description: Do not edit
'''
import os 
import gym
import random 
import numpy as np
from collections import deque
from keras.layers import Input,Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


class DQN:
    def __init__(self) -> None:
        self.model=self.build_model()
        self.target_model=self.build_model()
        self.update_target_model()
        if os.path.exists('dqn.h5'):
            self.model.load_weight('dqn.h5')
        # 经验池/样本池
        self.memory_buffer=deque(maxlen=2000)
        # Q_value的discount rate,以便来计算未来reward的折扣回报
        self.gamma=0.95
        # 贪婪选择法的随机选择行为的程度
        self.epsilon=1.0
        # 上述参数的衰减率
        self.epsilon_decay=0.995
        # 最小