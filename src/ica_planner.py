# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import copy
import time

from mmlda_bhmm.word_client import WordClient
from hmm_mmlda_hmm import ICA, df2list, make_obs

class Planner(object):
    """
    現在の状態と終了状態(言語命令)を受け取ることでプランニングを行う
    """
    def __init__(self):
        # パラメタ
        # word用
        self.word_client = WordClient()
        self.word_client.num_dim = 200

        self.model_episode = 0
        self.ica = ICA()
        self.ica.hmm_planning.h.output_dim = 6
        self.ica.load_model(self.model_episode)

    def plane(self, obs, sentence_instruction, step_n):
        """
        プランニングのメイン
        obs: 現在の観測情報
        sentence_instruction: 終了状態(言語命令)

        results list: プランニング結果
        [[object_hist, motion_hist, place_hist]
        ....]
        """
        # 初期状態の計算
        self.ica.recog(obs)
        start_z = self.ica.mlda_top.get_forward_msg()

        # 終了状態の計算
        obs4end = [None] * 4
        instruction_featuret = self.get_instruction_feature(sentence_instruction)
        obs4end[0] = np.zeros(obs[0].shape) # 未観測情報は頻度0
        obs4end[1] = np.zeros(obs[1].shape) # 未観測情報は頻度0
        obs4end[2] = np.zeros(obs[2].shape) # 未観測情報は頻度0
        obs4end[3] = np.array(instruction_featuret)
        self.ica.recog(obs4end)
        end_z = self.ica.mlda_top.get_forward_msg()

        # viterbi
        state_path = self.ica.hmm_planning.h.viterbi(np.argmax(start_z), np.argmax(end_z), step_n)
        # state_path = self.ica.hmm_planning.h.viterbi(31, 31, 1)
        result = []
        for state in state_path:
            result.append(self.state2action_2(state))
        return result
                        


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

        z1_sample = np.argmax(pwz_mlda_top[0][:,z4_sample])
        z2_sample = np.argmax(pwz_mlda_top[1][:,z4_sample])
        z3_sample = np.argmax(pwz_mlda_top[2][:,z4_sample])

        o1_sample = pwz_mlda1[0][:,z1_sample]
        o2_sample = pwz_mlda1[1][:,z1_sample]
        o3_sample = pwz_mlda2[0][:,z2_sample]
        o4_sample = pwz_mlda2[1][:,z2_sample]
        o5_sample = pwz_mlda3[0][:,z3_sample]
        o6_sample = pwz_mlda3[1][:,z3_sample]

        z1 = hmm_out_p.dot(pwz_mlda_top[0].T)
        z2 = hmm_out_p.dot(pwz_mlda_top[1].T)
        z3 = hmm_out_p.dot(pwz_mlda_top[2].T)

        o1 = z1.dot(pwz_mlda1[0].T)
        o2 = z1.dot(pwz_mlda1[1].T)
        o3 = z2.dot(pwz_mlda2[0].T)
        o4 = z2.dot(pwz_mlda2[1].T)
        o5 = z3.dot(pwz_mlda3[0].T)
        o6 = z3.dot(pwz_mlda3[1].T)

        return o1_sample, o2_sample, o3_sample, o4_sample, o5_sample, o6_sample
        return o1[state], o2[state], o3[state], o4[state], o5[state], o6[state]

    def get_instruction_feature(self,sen):
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

if __name__ == '__main__':
    planner = Planner()
    obs_df = pd.read_csv("./obs_data.csv", index_col=[0,1,2], header=[0,1] )
    sen_df = pd.read_csv("./sen_data.csv")
    while True:
        obs, correct = make_obs(obs_df, sen_df)
        O_O = obs[0][0].reshape(1,-1) # 物体情報
        M_M = obs[1][0].reshape(1,-1) # 動作情報
        P_P = obs[2][0].reshape(1,-1) # 場所情報
        W_ALL = obs[3][0].reshape(1,-1) #言語情報 (初期状態に言語情報を使用しない場合は0で埋める)
        obs4plane = [O_O, M_M, P_P, W_ALL]
        planing_results = planner.plane(obs4plane, "リビングで箱に入れる", step_n=1)
        print (planing_results)
        print ("end")
        break
