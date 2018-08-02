# -*- coding: utf-8 -*-
#!/usr/bin/env python
#%%
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import copy
import time

# gym
import gym
import gym_maze

from mmlda_bhmm.word_client import WordClient
from rl_hmm_mmlda_hmm import ICA

category_score = {0:4,1:5,2:6,3:3,4:8,5:7,6:2,7:1,8:0}
class SimulationLearning(object):
    """
    シミュレーション的に概念と方策を学習するプログラム
    ダミーデータ : joeさんデータ
    ダミー環境 : gym_maze
    """
    def __init__(self):
        # パラメタ
        self.env = gym.make('maze-sample-3x3-v0')
        self.is_render = True
        self.dim_maze = 3
        self.episode_num=200
        self.step_num=50

        # data 保存用
        self.dataframe = pd.DataFrame()
        self.sen_df = pd.DataFrame()

        # word用
        self.word_client = WordClient()
        self.word_client.num_dim = 200
        # self.word_client.load_codebook(save_dir + "/codebook.txt")

        self.model_episode = 199
        self.ica = ICA()
        self.ica.hmm_planning.h.output_dim = 32
        self.ica.load_model(self.model_episode)
        self.updata_itr = 5

        # load data
        JOE_DATA_DIR = "/home/miyazawa/catkin_ws/src/miyazawa_sim/miyazawa_main/src/joe_data"
        self.all_vision_hist = np.loadtxt(JOE_DATA_DIR + "/all_vision_hist.txt", delimiter="\t")
        self.all_audio_hist = np.loadtxt(JOE_DATA_DIR + "/all_audio_hist.txt", delimiter="\t")
        self.all_tactile_hist = np.loadtxt(JOE_DATA_DIR + "/all_tactile_hist.txt", delimiter="\t")
        self.all_words_hist = np.loadtxt(JOE_DATA_DIR + "/all_google_words_hist.txt", delimiter="\t")
        self.joe_correct_category = np.loadtxt(JOE_DATA_DIR + "/all_correct_category_number.txt")
        self.sen_miya = pd.read_csv("/home/miyazawa/catkin_ws/src/miyazawa_sim/miyazawa_main/src/joe_data/sentences_miyazawa_anatetion.csv")

        
    def simulate(self):
        """
        メインのループ
        """
        action_path = None
        # episode
        for episode in range(self.episode_num):
            observation = self.env.reset() 
            
            # step
            for step in range(self.step_num):
                print ("episode:{0}, step:{1}".format(episode, step))
                if self.is_render is True:
                    self.env.render()
                old_observation = copy.deepcopy(observation)
                
                # calc state 
                if action_path is None:
                    action = int(raw_input("action set:"))
                else:
                    try:
                        action = action_path.pop(0)
                        time.sleep(0.5)
                    except:
                        raw_input("end")
                        action_path = None

                # action = np.random.randint(4)
                observation, reward, done, info = self.env.step(action)
                old_object_category = int(old_observation[0] + (old_observation[1] * self.dim_maze))
                object_category = observation[0] + (observation[1] * self.dim_maze)
                
                # calc reward
                old_score = category_score[old_object_category]
                new_score = category_score[object_category]
                reward = old_score - new_score

                # add data to dataframe
                self.add_data(old_object_category, action, reward, episode, step)
                self.add_sentence(old_object_category, action, reward, episode, step)
                
                # make obs data
                obs, correct = self.make_obs(episode,step)
                # recog , startのzを計算
                if action_path is None:
                    s_yn = raw_input("Is This start point? [y/n]")
                    if s_yn is "y":
                        # all obs
                        self.ica.recog(obs,correct)
                        start_z = self.ica.mlda_top.get_forward_msg()
                        # word obs
                        instruction_featuret = self.get_instruction_feature()
                        for i in range(6):
                            obs[i] = np.zeros(obs[i].shape)
                        obs[5] = np.array(instruction_featuret)
                        obs[6] = None
                        obs[7] = None
                        self.ica.recog(obs,correct)
                        end_z = self.ica.mlda_top.get_forward_msg()
                        # viterbi
                        step_num = int(raw_input("step num:"))
                        state_path = self.ica.hmm_planning.h.viterbi(np.argmax(start_z), np.argmax(end_z), step_num)
                        # state_path = self.ica.hmm_planning.h.viterbi(31, 31, 1)
                        tmp_list = []
                        for state in state_path:
                            tmp_list.append(self.state2action_2(state))
                        # print start_z, end_z
                        print state_path
                        print tmp_list
                        action_path = tmp_list[1:]
                        


    def state2action(self,state): 
        def calc_pwz(n_mz, n_mzw, m):
            __beta = 1.0 
            dim = len(n_mzw[m])
            Pwz = (n_mzw[m].T + __beta) / (n_mz[m] + dim *__beta)
            return Pwz

        pwz_mlda1 = []
        n_mz = self.ica.mlda1._MLDA__n_mz
        n_mzw = self.ica.mlda1._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda1.append( calc_pwz(n_mz,n_mzw,m) )

        pwz_mlda2 = []
        n_mz = self.ica.mlda2._MLDA__n_mz
        n_mzw = self.ica.mlda2._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda2.append( calc_pwz(n_mz,n_mzw,m) )

        pwz_mlda3 = []
        n_mz = self.ica.mlda3._MLDA__n_mz
        n_mzw = self.ica.mlda3._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda3.append( calc_pwz(n_mz,n_mzw,m) )

        pwz_mlda_top = []
        n_mz = self.ica.mlda_top._MLDA__n_mz
        n_mzw = self.ica.mlda_top._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda_top.append( calc_pwz(n_mz,n_mzw,m) ) 

        hmm_out_p = self.ica.hmm_planning.h.calc_all_outpu_prob()

        z1 = hmm_out_p.dot(pwz_mlda_top[0].T)
        z2 = hmm_out_p.dot(pwz_mlda_top[1].T)
        z3 = hmm_out_p.dot(pwz_mlda_top[2].T)

        o1 = z1.dot(pwz_mlda1[0].T)
        o2 = z1.dot(pwz_mlda1[1].T)
        o3 = z1.dot(pwz_mlda1[2].T)
        o4 = z1.dot(pwz_mlda1[3].T)
        o5 = z2.dot(pwz_mlda2[0].T)
        o6 = z2.dot(pwz_mlda2[1].T)
        o7 = z3.dot(pwz_mlda3[0].T)
        o8 = z3.dot(pwz_mlda3[1].T)

        print o5[state], len(o5[state])
        return np.argmax(o5[state])


    def state2action_2(self,state): 
        def calc_pwz(n_mz, n_mzw, m):
            __beta = 1.0 
            dim = len(n_mzw[m])
            Pwz = (n_mzw[m].T + __beta) / (n_mz[m] + dim *__beta)
            return Pwz

        pwz_mlda1 = []
        n_mz = self.ica.mlda1._MLDA__n_mz
        n_mzw = self.ica.mlda1._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda1.append( calc_pwz(n_mz,n_mzw,m) )

        pwz_mlda2 = []
        n_mz = self.ica.mlda2._MLDA__n_mz
        n_mzw = self.ica.mlda2._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda2.append( calc_pwz(n_mz,n_mzw,m) )

        pwz_mlda3 = []
        n_mz = self.ica.mlda3._MLDA__n_mz
        n_mzw = self.ica.mlda3._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda3.append( calc_pwz(n_mz,n_mzw,m) )

        pwz_mlda_top = []
        n_mz = self.ica.mlda_top._MLDA__n_mz
        n_mzw = self.ica.mlda_top._MLDA__n_mzw
        for m in range(len(n_mz)):
            pwz_mlda_top.append( calc_pwz(n_mz,n_mzw,m) ) 

        hmm_out_p = self.ica.hmm_planning.h.calc_all_outpu_prob()

        z4_sample = np.argmax(hmm_out_p[state])
        z2_sample = np.argmax(pwz_mlda_top[1][:,z4_sample])
        o5_sample = np.argmax(pwz_mlda2[0][:,z2_sample])

        z1 = hmm_out_p.dot(pwz_mlda_top[0].T)
        z2 = hmm_out_p.dot(pwz_mlda_top[1].T)
        z3 = hmm_out_p.dot(pwz_mlda_top[2].T)

        o1 = z1.dot(pwz_mlda1[0].T)
        o2 = z1.dot(pwz_mlda1[1].T)
        o3 = z1.dot(pwz_mlda1[2].T)
        o4 = z1.dot(pwz_mlda1[3].T)
        o5 = z2.dot(pwz_mlda2[0].T)
        o6 = z2.dot(pwz_mlda2[1].T)
        o7 = z3.dot(pwz_mlda3[0].T)
        o8 = z3.dot(pwz_mlda3[1].T)

        print pwz_mlda2[0][z2_sample]
        return o5_sample

    def get_instruction_feature(self,):
        sen = raw_input("sentences: ")
        self.word_client.load_codebook("./codebook.txt")
        self.word_client.setAccept()
        self.word_client.sentence_data_cb(sen)
        self.word_client.setReject()
        self.word_client.split_sentence()
        self.word_client.update_word_codebook()
        self.word_client.words2histogram()
        splited_sen = self.word_client.get_splited_sentences()
        word_histogram = self.word_client.get_histgram()
        self.word_client.reset()
        return np.array([word_histogram])*100

    def add_data(self,state, action, reward, episode, step):
        """
        episode: データのエピソード
        """
        # objectの選択
        candidate_objects = np.where(self.joe_correct_category==state)[0]
        object_num = np.random.choice(candidate_objects)

        state_pd = pd.DataFrame([state]) 
        action_pd = pd.DataFrame([action])
        reward_pd = pd.DataFrame([reward])
        object_pd = pd.DataFrame([object_num])
        
        tmp_pd = pd.concat([state_pd,action_pd,reward_pd,object_pd],axis=1)
        tmp_pd.columns = ["state","action","reward","object_num"]
        tmp_pd["step"] = step
        tmp_pd["episode"] = episode
        
        self.dataframe = pd.concat([self.dataframe, tmp_pd],ignore_index=True)
        return True

    def add_sentence(self,state, action, reward, episode, step):
        # 文章の追加
        tmp_df = self.sen_miya[(self.sen_miya.category == state) & (self.sen_miya.action==action)].sample(n=5)
        tmp_df["episode"] = episode
        tmp_df["step"] = step
        tmp_df = tmp_df.drop(["category","action","reward"],axis=1)
        self.sen_df = pd.concat([self.sen_df, tmp_df],ignore_index=True)
        return True
        # sentences_in_maze.to_csv("./maze_sentences_data.csv",index=False)

    def make_obs(self, episode, step):
        # データの成形
        splited_sentences = []
        
        vision_object = []
        audio_object = []
        tactile_object = []
        words_all = []

        motion_motion = []
        reward_reward = []
        
        
        if step is None: # 1 episode分
            sen_df = self.sen_df[self.sen_df.episode==episode]
            dataframe = self.dataframe[self.dataframe.episode == episode]
        else: # 1 step分
            sen_df = self.sen_df[(self.sen_df.episode==episode) & (self.sen_df.step==step)]
            dataframe = self.dataframe[(self.dataframe.episode == episode) & (self.dataframe.step == step)]           

        category_list = dataframe.state.values
        action_list = dataframe.action.values
        reward_list = dataframe.reward.values
        object_list = dataframe.object_num.values
            
        
        # ステップ数分データを貯める
        e = episode
        for s in range(len(category_list)):
            category = category_list[s]
            action = action_list[s]
            reward = reward_list[s]

            # candidate_objects = np.where(self.joe_correct_category==category)[0]
            # object_num = np.random.choice(candidate_objects)
            object_num = object_list[s]

            vision_object.append(self.all_vision_hist[object_num])
            audio_object.append(self.all_audio_hist[object_num])
            tactile_object.append(self.all_tactile_hist[object_num])
            

            motion_motion.append(np.eye(4)[action])
            reward_reward.append(np.eye(3)[reward+1])
        
            # word の処理
            self.word_client.setAccept()
            for sen in sen_df[(sen_df.episode==e) & (sen_df.step==s)].sentences.values:
                self.word_client.sentence_data_cb(sen)
            self.word_client.setReject()
            self.word_client.split_sentence()
            self.word_client.update_word_codebook()
            self.word_client.words2histogram()
            splited_sen = self.word_client.get_splited_sentences()
            word_histogram = self.word_client.get_histgram()
            self.word_client.reset()
            
            words_all.append(word_histogram)
            splited_sentences.extend(splited_sen)
        
        codebook = self.word_client._codebook
        obs = [np.array(vision_object), np.array(audio_object), np.array(tactile_object), 
        np.array(motion_motion), np.array(reward_reward), np.array(words_all), 
        np.array(splited_sentences), codebook]
        correct = [dataframe.state.values, dataframe.action.values, dataframe.reward.values]
        return obs, correct

if __name__ == '__main__':
    SL = SimulationLearning()
    SL.simulate()
    # SL.dataframe.to_csv("./maze_episode_data.csv",index=False)
    # SL.sen_df.to_csv("./maze_sen.csv",index=False)
    print ("end")



