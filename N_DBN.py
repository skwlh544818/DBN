'''
Author: sunkaiwei
Date: 2021-04-04 16:49:40
LastEditTime: 2021-04-04 20:43:49
LastEditer: sunkaiwei
Description: Do not edit
'''
import os 
import gym
import random
from keras.losses import hinge 
import numpy as np
from collections import deque
from keras.layers import Input,Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from numpy.lib.histograms import histogram
from tensorflow.python.ops.gen_random_ops import random_gamma_eager_fallback


class DQN:
    def __init__(self) -> None:
        self.model=self.build_model()
        self.target_model=self.build_model()
        self.update_target_model()
        if os.path.exists('dqn.h5'):
            self.model.load_weights('dqn.h5')
        # 经验池/样本池
        self.memory_buffer=deque(maxlen=2000)
        # Q_value的discount rate,以便来计算未来reward的折扣回报
        self.gamma=0.95
        # 贪婪选择法的随机选择行为的程度
        self.epsilon=1.0
        # 上述参数的衰减率
        self.epsilon_decay=0.995
        # 最小随机探索概率
        self.epsilon_min=0.01
        self.env=gym.make('CartPole-v0')
    def build_model(self):
        '''基本网络结构'''
        inputs=Input(shape=(4,))
        x=Dense(16,activation='relu')(inputs)
        x=Dense(16,activation='relu')(x)
        x=Dense(2,activation='linear')(x)
        model=Model(inputs=inputs,outputs=x)
        return model
    def update_target_model(self):
        '''更新target_model'''
        self.target_model.set_weights(self.model.get_weights())
    def egreedy_action(self,state):
        '''ε-greedy选择action
        state:状态
        action:动作
        '''
        if np.random.random()<=self.epsilon:
            return random.randint(0,1)
        else:
            q_values=self.model.predict(state)[0]
            return np.argmax(q_values)
    def remember(self,state,action,reward,next_state,done):
        '''向经验池添加数据
        state:状态
        action:动作
        reward:回报
        next_state:下一个状态
        done:游戏结束标志
        '''
        item=(state,action,reward,next_state,done)
        self.memory_buffer.append(item)
    def update_epsilon(self):
        '''更新epsilon
        '''
        if self.epsilon>=self.epsilon_min:
            self.epsilon*=self.epsilon_decay
    def process_batch(self,batch):
        '''batch数据处理
        batech:batch size
        x:states
        y:[Q_value1,Q_value2]
        '''
        # 从经验池中随机采样一个batch
        data=random.sample(self.memory_buffer,batch)
        # 生成Q_target
        states=np.array([d[0] for d in data])
        next_states=np.array([d[3] for d in data])
        y=self.model.predict(states)
        q=self.target_model.predict(next_states)
        for i,(_,action,reward,_,done) in enumerate(data):
            target=reward
            if not done:
                target +=self.gamma*np.amax(q[i])
            y[i][action]=target
        return states,y
    def train(self,episode,batch):
        '''训练
        episode:游戏次数
        batch:batch size
        history:训练记录
        '''
        self.model.compile(loss='mse',optimizer=Adam(1e-3))
        history={'episode':[],'Episode_reward':[],'Loss':[]}
        count=0
        for i in range(episode):
            observation=self.env.reset()
            reward_sum=0
            loss=np.infty
            done=False
            while not done:
                # 通过贪婪选择法ε-greedy选择action
                x=observation.reshape(-1,4)
                action=self.egreedy_action(x)
                observation,reward,done,_=self.env.step(action)
                reward_sum+=reward
                # 将数据加入到经验池
                self.remember(x[0],action,reward,observation,done)
                if len(self.memory_buffer)>batch:
                    # 训练
                    X,y=self.process_batch(batch)
                    loss=self.model.train_on_batch(X,y)
                    count+=1
                    # 减少egreedy的epsilon参数
                    self.update_epsilon()
                    # 固定次数更新target——model
                    if count != 0 and count%20==0:
                        self.update_target_model()
            if i%5==0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)
                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i,reward_sum,loss,self.epsilon))
        self.model.save_weights('dqn.h5')
        return history
    def play(self):
        observation=self.env.reset()
        count=0
        reward_sum=0
        random_episode=0
        while random_episode<10:
            self.env.render()
            x=observation.reshape(-1,4)
            q_values=self.model.predict(x)[0]
            action=np.argmax(q_values)
            observation,reward,done,_=self.env.step(action)
            count+=1
            reward_sum+=reward
            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum,count))
                random_episode+=1
                reward_sum=0
                count=0
                observation=self.env.reset()
        self.env.close()
if __name__=='__main__':
    model=DQN()
    history=model.train(600,32)
    model.play()
    
        
                

        