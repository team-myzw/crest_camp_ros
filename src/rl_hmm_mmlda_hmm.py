# -*- coding: utf-8 -*-
#!/usr/bin/env python
#%%
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import copy

# gym
import gym
import gym_maze

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

        # load data
        JOE_DATA_DIR = "/home/miyazawa/catkin_ws/src/miyazawa_sim/miyazawa_main/src/joe_data"
        self.all_vision_hist = np.loadtxt(JOE_DATA_DIR + "/all_vision_hist.txt", delimiter="\t")
        self.all_audio_hist = np.loadtxt(JOE_DATA_DIR + "/all_audio_hist.txt", delimiter="\t")
        self.all_tactile_hist = np.loadtxt(JOE_DATA_DIR + "/all_tactile_hist.txt", delimiter="\t")
        self.all_words_hist = np.loadtxt(JOE_DATA_DIR + "/all_google_words_hist.txt", delimiter="\t")
        self.joe_correct_category = np.loadtxt(JOE_DATA_DIR + "/all_correct_category_number.txt")
        self.sen_miya = pd.read_csv("/home/miyazawa/catkin_ws/src/miyazawa_sim/miyazawa_main/src/joe_data/sentences_miyazawa_anatetion.csv")
 
        # モデル
        self.ica = ICA()
        
    def simulate(self):
        """
        メインのループ
        """
        # episode
        for episode in range(self.episode_num):
            observation = self.env.reset() 
            self.next_action = np.random.randint(4) # エピソードの最初のステップはランダムで動く
            
            # step
            for step in range(self.step_num):
                print ("episode:{0}, step:{1}".format(episode, step))
                if self.is_render is True:
                    self.env.render()
                old_observation = copy.deepcopy(observation)
                
                # calc state 
                action = self.next_action
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
                # recog
                self.next_action = self.ica.recog(obs,correct)

                if done:
                    print("Episode finished after {} timesteps".format(step+1))
                    obs, correct = self.make_obs(episode,None)
                    self.ica.leran(obs,correct)
                    break
            else: # for文に対してのelse step_numでepisodeを終了できなかった場合
                print("Episode not finished after {} timesteps".format(step+1))
                obs, correct = self.make_obs(episode,None)
                self.ica.leran(obs,correct)
                
            self.dataframe.to_csv("./maze_episode_data.csv",index=False)
            self.sen_df.to_csv("./maze_sen.csv",index=False)


                
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
        
        self.word_client.dump_codebook("./codebook.txt")
        codebook = self.word_client._codebook
        obs = [np.array(vision_object), np.array(audio_object), np.array(tactile_object), 
        np.array(motion_motion), np.array(reward_reward), np.array(words_all), 
        np.array(splited_sentences), codebook]
        correct = [dataframe.state.values, dataframe.action.values, dataframe.reward.values]
        return obs, correct

class ICA(object):
    """
    icaモデル
    """
    def __init__(self):
        # モデル全体の繰り返し回数
        self.updata_itr = 5
        # mldaの重み
        w = 500
        # 各モジュールの繰り返し回数
        itration = 50
        itration_recog = 10
        # モジュールの定義
        self.obs1 = srk.Observation( None )    # 視覚情報
        self.obs2 = srk.Observation( None )    # 聴覚情報
        self.obs3 = srk.Observation( None )    # 触覚情報
        self.obs4 = srk.Observation( None )    # 物体単語
        self.obs5 = srk.Observation( None )    # 動作情報
        self.obs6 = srk.Observation( None )    # 動作単語
        self.obs7 = srk.Observation( None )    # 報酬情報
        self.obs8 = srk.Observation( None )    # 報酬単語

        self.mlda1 = mlda.MLDA(8, [w, w, w, w], itr=itration, itr_recog=itration_recog)
        self.mlda2 = mlda.MLDA(4, [w, w], itr=itration, itr_recog=itration_recog)
        self.mlda3 = mlda.MLDA(3, [w, w], itr=itration, itr_recog=itration_recog)
        self.mlda_top = mlda.MLDA(32, [w, w, w], itr=itration, itr_recog=itration_recog)
        self.hmm_planning = hmm.HMM(32, itr=itration,name="hmm_planning", itr_recog=itration_recog, multinomial_dim=10)
        self.hmm_launguage = mmlda_bhmm.MMLDA_BHMM(itration,3,13,0.008,150)
        self.reinforce_ = reinforce.REINFORCE(32, 4, 0.01)

    def leran(self,obs,correct):
        # learn
        V_O = obs[0]
        A_O = obs[1]
        T_O = obs[2]
        M_M = obs[3]
        R_R = obs[4]

        W_ALL = obs[5]
        
        sentences = obs[6]
        codebook = obs[7]

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
        self.reinforce_.reset_msg()
        
        # update obs
        #各概念の単語情報を作成
        W_O, W_M, W_P = self.hmm_launguage.makeHist(W_ALL,)
        
        self.obs1.set_forward_msg(V_O )     # 視覚情報
        self.obs2.set_forward_msg(A_O )     # 聴覚情報
        self.obs3.set_forward_msg(T_O )     # 触覚情報
        self.obs4.set_forward_msg(W_O)      # 物体単語

        self.obs5.set_forward_msg(M_M )     # 動作情報
        self.obs6.set_forward_msg(W_M)      # 動作単語

        self.obs7.set_forward_msg(R_R )     # 報酬情報
        self.obs8.set_forward_msg(W_P)      # 報酬単語

        # connect
        self.mlda1.connect( self.obs1, self.obs2, self.obs3, self.obs4 )
        self.mlda2.connect( self.obs5, self.obs6 )
        self.mlda3.connect( self.obs7, self.obs8 )
        self.mlda_top.connect( self.mlda1, self.mlda2, self.mlda3 )
        self.hmm_planning.connect(self.mlda_top)
        self.reinforce_.connect(self.hmm_planning, self.obs5, self.obs7)

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
            self.reinforce_.update()
        
        n_mwz_obj, _ = mlda_p.load_model("module008_mlda/{0:03d}/".format(self.mlda1._MLDA__nit-1))
        n_mwz_mot, _ = mlda_p.load_model("module009_mlda/{0:03d}/".format(self.mlda2._MLDA__nit-1))
        n_mwz_pla, _ = mlda_p.load_model("module010_mlda/{0:03d}/".format(self.mlda3._MLDA__nit-1))
        #相互情報量の計算
        self.hmm_launguage.set_codebook(codebook)
        self.hmm_launguage.calc_mi([n_mwz_obj[3].T, n_mwz_mot[1].T, n_mwz_pla[1].T])
        # 文法の学習
        self.hmm_launguage.update(sentences,online_flag)

        # # 言語情報の重み付け
        # W_O, W_M, W_P = self.hmm_launguage.makeHist(W_ALL)
        # obs4.set_forward_msg(W_O)
        # obs6.set_forward_msg(W_M)
        # obs8.set_forward_msg(W_P)

        # パラメータの忘却
        beta = 0.8
        self.mlda1.reduction(beta)
        self.mlda2.reduction(beta)
        self.mlda3.reduction(beta)
        self.mlda_top.reduction(beta)
        self.hmm_planning.reduction(beta)
        self.hmm_launguage.reduction(beta)
        
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
       
        nit = episode # 更新回数がupdate分少ないため
        self.hmm_launguage.load_model( os.path.join( self.hmm_launguage.get_name(), "model_%03d.pickle" % nit) )
        self.hmm_launguage.set_nit(nit)
        self.reinforce_.load_model( os.path.join( self.reinforce_.get_name(), "rl_param_%03d.txt" % nit) )
        self.reinforce_.set_nit(nit)

    def recog(self,obs,correct):
        # Recog


        # reset msg
        self.mlda1.reset_msg()
        self.mlda2.reset_msg()
        self.mlda3.reset_msg()
        self.mlda_top.reset_msg()
        self.hmm_planning.reset_msg()
        self.reinforce_.reset_msg()

        V_O = obs[0]
        A_O = obs[1]
        T_O = obs[2]
        M_M = obs[3]
        R_R = obs[4]

        W_ALL = obs[5]

        # 正解ラベルを入れておくと精度出してくれる recogは使わない
        self.mlda1._MLDA__category = None
        self.mlda2._MLDA__category = None
        self.mlda3._MLDA__category = None

        # update obs
        #各概念の単語情報を作成
        W_O, W_M, W_P = self.hmm_launguage.makeHist(W_ALL,)

        self.obs1.set_forward_msg(V_O )     # 視覚情報
        self.obs2.set_forward_msg(A_O )     # 聴覚情報
        self.obs3.set_forward_msg(T_O )     # 触覚情報
        self.obs4.set_forward_msg(W_O)      # 物体単語

        self.obs5.set_forward_msg(M_M )     # 動作情報
        self.obs6.set_forward_msg(W_M)      # 動作単語

        self.obs7.set_forward_msg(R_R )     # 報酬情報
        self.obs8.set_forward_msg(W_P)      # 報酬単語

        # connect
        self.mlda1.connect( self.obs1, self.obs2, self.obs3, self.obs4 )
        self.mlda2.connect( self.obs5, self.obs6 )
        self.mlda3.connect( self.obs7, self.obs8 )
        self.mlda_top.connect( self.mlda1, self.mlda2, self.mlda3 )
        self.hmm_planning.connect(self.mlda_top)
        self.reinforce_.connect(self.hmm_planning, self.obs5, self.obs7)

        # 認識メインループ
        online_flag = False
        for i in range(self.updata_itr):
            # print ("mmlda + bhmm repeat: {0}".format(i))
            # update model
            self.mlda1.update(online_flag, True)
            self.mlda2.update(online_flag, True)
            self.mlda3.update(online_flag, True)
            self.mlda_top.update(online_flag, True)
            self.hmm_planning.update(online_flag, True)

        # 行動選択
        action = self.reinforce_.update(True)
        return action

if __name__ == '__main__':
    SL = SimulationLearning()
    SL.simulate()
    SL.dataframe.to_csv("./maze_episode_data.csv",index=False)
    SL.sen_df.to_csv("./maze_sen.csv",index=False)
    print ("end")



