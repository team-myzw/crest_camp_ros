# -*- coding: utf-8 -*-
#!/usr/bin/env python
#%%
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import copy

# serket modules
import serket as srk
import mlda
from mlda import mlda as mlda_p
import hmm
import reinforce
import mmlda_bhmm
from mmlda_bhmm.word_client import WordClient
from mmlda_bhmm import  WordsInference
from mmlda_bhmm import  bhmm


def df2list(tmp_df, hist_name):
    tmp_hists = []
    tmp_df = tmp_df.loc[:,hist_name]
    for i,row in tmp_df.iterrows():
        tmp_hists.append(row.values)
    return np.array(tmp_hists)

def make_obs(obs_data, sen_data):
    # データの成形
    word_client = WordClient()
    object_object = df2list(obs_data,"category_hist")
    motion_motion = df2list(obs_data,"action_hist")
    place_place = df2list(obs_data,"place_hist")
    words_all =  df2list(obs_df,"word_hist")  
    splited_sentences = []    

    # 全sentenceの処理
    word_client.setAccept()
    for _,sen in sen_data.iterrows():
        word_client.sentence_data_cb(sen.sentence)
    word_client.setReject()
    word_client.split_sentence()
    word_client.update_word_codebook()
    word_client.words2histogram()
    splited_sen = word_client.get_splited_sentences()
    word_histogram = word_client.get_histgram()
    word_client.reset()
    splited_sentences.extend(splited_sen)
    codebook = word_client._codebook
    word_client.dump_codebook("./codebook.txt")

    obs = [object_object, motion_motion,
    place_place, words_all, 
    np.array(splited_sentences), codebook]
    correct = [None,None,None]
    # correct = [dataframe.state.values, dataframe.action.values, dataframe.place.values]
    return obs, correct

class ICA(object):
    """
    icaモデル
    """
    def __init__(self):
        # カテゴリ数
        object_category_num = 3
        motion_category_num = 3
        place_category_num = 3
        top_category_num = 6
        hmm_category_num = 6

        # 言語学習用
        bottom_concept_num = 3
        particle_num = 3 # 助詞の種類
        word_threshold = 0.08
        word_max_dim = 200

        # モデル全体の繰り返し回数
        self.updata_itr = 5
        # mldaの重み
        w = 500
        # 各モジュールの繰り返し回数
        itration = 50
        itration_recog = 10
        # モジュールの定義
        self.obs1 = srk.Observation( None )    # 物体情報
        self.obs2 = srk.Observation( None )    # 物体単語
        self.obs3 = srk.Observation( None )    # 動作情報
        self.obs4 = srk.Observation( None )    # 動作単語
        self.obs5 = srk.Observation( None )    # 場所情報
        self.obs6 = srk.Observation( None )    # 場所単語

        self.mlda1 = mlda.MLDA(object_category_num, [w, w, w, w], itr=itration, itr_recog=itration_recog)
        self.mlda2 = mlda.MLDA(motion_category_num, [w, w], itr=itration, itr_recog=itration_recog)
        self.mlda3 = mlda.MLDA(place_category_num, [w, w], itr=itration, itr_recog=itration_recog)
        self.mlda_top = mlda.MLDA(top_category_num, [w, w, w], itr=itration, itr_recog=itration_recog)
        self.hmm_planning = hmm.HMM(hmm_category_num, itr=itration,name="hmm_planning", itr_recog=itration_recog, multinomial_dim=10)
        self.hmm_launguage = mmlda_bhmm.MMLDA_BHMM(itration,bottom_concept_num, 
        bottom_concept_num + particle_num, word_threshold ,word_max_dim)
        # self.reinforce_ = reinforce.REINFORCE(32, 4, 0.01)

    def leran(self,obs,correct):
        # learn
        O_O = obs[0]
        M_M = obs[1]
        P_P = obs[2]

        W_ALL = obs[3]
        
        sentences = obs[4]
        codebook = obs[5]

        # 正解ラベルを入れておくと精度出してくれる
        self.mlda1._MLDA__category = correct[0]
        self.mlda2._MLDA__category = correct[1]
        self.mlda3._MLDA__category = correct[2]

        # reset msg
        self.mlda1.reset_msg()
        self.mlda2.reset_msg()
        self.mlda3.reset_msg()
        self.mlda_top.reset_msg()
        self.hmm_planning.reset_msg()
        # self.reinforce_.reset_msg()
        
        # update obs
        #各概念の単語情報を作成
        W_O, W_M, W_P = self.hmm_launguage.makeHist(W_ALL,)
        
        self.obs1.set_forward_msg(O_O)     # 物体情報
        self.obs2.set_forward_msg(W_O)      # 物体単語

        self.obs3.set_forward_msg(M_M )     # 動作情報
        self.obs4.set_forward_msg(W_M)      # 動作単語

        self.obs5.set_forward_msg(P_P )     # 場所情報
        self.obs6.set_forward_msg(W_P)      # 場所単語

        # connect
        self.mlda1.connect( self.obs1, self.obs2 )
        self.mlda2.connect( self.obs3, self.obs4 )
        self.mlda3.connect( self.obs5, self.obs6 )
        self.mlda_top.connect( self.mlda1, self.mlda2, self.mlda3 )
        self.hmm_planning.connect(self.mlda_top)
        # self.reinforce_.connect(self.hmm_planning, self.obs5, self.obs7)

        # 学習メインループ
        for i in range(self.updata_itr):
            print ("learn each modules repeat: {0}/{1}".format(i+1,self.updata_itr))
            online_flag = (i is self.updata_itr-1)
            # update model
            self.mlda1.update(online_flag)
            self.mlda2.update(online_flag)
            self.mlda3.update(online_flag)
            self.mlda_top.update(online_flag)
            self.hmm_planning.update(online_flag)
            # self.reinforce_.update()
            n_mwz_obj = self.mlda1.n_mzw4bhmm
            n_mwz_mot = self.mlda2.n_mzw4bhmm
            n_mwz_pla = self.mlda3.n_mzw4bhmm
            #相互情報量の計算
            self.hmm_launguage.set_codebook(codebook)
            self.hmm_launguage.calc_mi([n_mwz_obj[1].T, n_mwz_mot[1].T, n_mwz_pla[1].T])
            # 文法の学習
            self.hmm_launguage.update(sentences,online_flag)

            # 言語情報の重み付け
            W_O, W_M, W_P = self.hmm_launguage.makeHist(W_ALL)
            self.obs2.set_forward_msg(W_O)
            self.obs4.set_forward_msg(W_M)
            self.obs6.set_forward_msg(W_P)

        # パラメータの忘却
        # beta = 0.8
        # self.mlda1.reduction(beta)
        # self.mlda2.reduction(beta)
        # self.mlda3.reduction(beta)
        # self.mlda_top.reduction(beta)
        # self.hmm_planning.reduction(beta)
        # self.hmm_launguage.reduction(beta)
        
    def load_model(self, episode):
        # モデルのロード
        # print ("load model episode: {0}".format(episode)) 
        nit = (episode+1) * self.updata_itr -1
        self.mlda1.load_model( os.path.join( self.mlda1.get_name(), "model_%03d.pickle" % nit) )
        self.mlda1.set_nit(nit)
        self.mlda2.load_model( os.path.join( self.mlda2.get_name(), "model_%03d.pickle" % nit) )
        self.mlda2.set_nit(nit)
        self.mlda3.load_model( os.path.join( self.mlda3.get_name(), "model_%03d.pickle" % nit) )
        self.mlda3.set_nit(nit)
        self.mlda_top.load_model( os.path.join( self.mlda_top.get_name(), "model_%03d.pickle" % nit) )
        self.mlda_top.set_nit(nit)
        self.hmm_planning.load_model( os.path.join( self.hmm_planning.get_name(), "model_%03d.pickle" % nit) )
        self.hmm_planning.set_nit(nit)
       
        # nit = episode # 更新回数がupdate分少ないため
        self.hmm_launguage.load_model( os.path.join( self.hmm_launguage.get_name(), "model_%03d.pickle" % nit) )
        self.hmm_launguage.set_nit(nit)
        # self.reinforce_.load_model( os.path.join( self.reinforce_.get_name(), "rl_param_%03d.txt" % nit) )
        # self.reinforce_.set_nit(nit)

    def recog(self,obs,correct):
        # Recog
        # reset msg
        self.mlda1.reset_msg()
        self.mlda2.reset_msg()
        self.mlda3.reset_msg()
        self.mlda_top.reset_msg()
        self.hmm_planning.reset_msg()
        # self.reinforce_.reset_msg()

        O_O = obs[0]
        M_M = obs[1]
        P_P = obs[2]

        W_ALL = obs[3]

        # 正解ラベルを入れておくと精度出してくれる recogは使わない
        self.mlda1._MLDA__category = None
        self.mlda2._MLDA__category = None
        self.mlda3._MLDA__category = None

        # update obs
        #各概念の単語情報を作成
        W_O, W_M, W_P = self.hmm_launguage.makeHist(W_ALL,)

        self.obs1.set_forward_msg(O_O )     # 物体情報
        self.obs2.set_forward_msg(W_O)      # 物体単語

        self.obs3.set_forward_msg(M_M )     # 動作情報
        self.obs4.set_forward_msg(W_M)      # 動作単語

        self.obs5.set_forward_msg(P_P )     # 場所情報
        self.obs6.set_forward_msg(W_P)      # 場所単語

        # connect
        self.mlda1.connect( self.obs1, self.obs2 )
        self.mlda2.connect( self.obs3, self.obs4 )
        self.mlda3.connect( self.obs5, self.obs6 )
        self.mlda_top.connect( self.mlda1, self.mlda2, self.mlda3 )
        self.hmm_planning.connect(self.mlda_top)
        # self.reinforce_.connect(self.hmm_planning, self.obs5, self.obs7)

        # 認識メインループ
        online_flag = False
        for i in range(self.updata_itr):
            # print ("mmlda + bhmm repeat: {0}".format(i))
            # update model
            self.mlda1.update(online_flag, True)
            self.mlda2.update(online_flag, True)
            self.mlda3.update(online_flag, True)
            self.mlda_top.update(online_flag, True)
            # self.hmm_planning.update(online_flag, True)

        # 行動選択
        # action = self.reinforce_.update(True)
        return True

if __name__ == '__main__':
    obs_df = pd.read_csv("./obs_data.csv", index_col=[0,1,2], header=[0,1] )
    sen_df = pd.read_csv("./sen_data.csv")
    obs, correct = make_obs(obs_df, sen_df)
    ica = ICA()
    ica.leran(obs,correct)
    print ("end")



