# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import shutil
import os
import bhmm
import WordsInference
import WordHist
import serket as srk
import mlda
from mlda import mlda as mlda_p
import threading

def Learn_srk():
    """
    mmlda と bhmmの相互作用による概念・言語・語彙の相互学習
    """

    #パラメータ
    ###############################################
    numOfRepeat = 3        #繰り返し学習する回数
    numOfAllConc = 5        #助詞を含んだ概念クラスの数
    numOfConc = 3           #助詞を除いた概念クラスの数
    threshold = 0.0001       #相互情報量の閾値
    # threshold = 0.008       #相互情報量の閾値
    ################################################

    V_O = np.loadtxt("LearnData/vision.Object")
    M_M = np.loadtxt("LearnData/motion.Motion")
    P_P = np.loadtxt("LearnData/place.Place")

    W_ALL = np.loadtxt("LearnData/words.All")

    # 正解ラベルを入れておくと精度出してくれる
    object_category = np.loadtxt( "LearnData/CorrectObject.txt" )
    motion_category = np.loadtxt( "LearnData/CorrectMotion.txt" )
    place_category = np.loadtxt( "LearnData/CorrectPlace.txt" )

    obs1 = srk.Observation( V_O )     # 物体情報
    obs2 = srk.Observation( W_ALL )     # 物体単語
    obs3 = srk.Observation( M_M )     # 動作情報
    obs4 = srk.Observation( W_ALL )     # 動作単語
    obs5 = srk.Observation( P_P )      # 場所情報
    obs6 = srk.Observation( W_ALL )       # 場所単語

    # mmldaの重み
    w = 100
    mlda1 = mlda.MLDA(5, [w, w], category=object_category)
    mlda2 = mlda.MLDA(5, [w, w], category=motion_category)
    mlda3 = mlda.MLDA(4, [w, w], category=place_category)
    mlda_top = mlda.MLDA(6, [w, w, w])

    mlda1.connect( obs1, obs2 )
    mlda2.connect( obs3, obs4 )
    mlda3.connect( obs5, obs6 )
    mlda_top.connect( mlda1, mlda2, mlda3 )

    for i in range(numOfRepeat):
        print ("mmlda + bhmm repeat: {0}".format(i))

        #各概念の単語情報を作成
        W_O, W_M, W_P = WordHist.makeHist(W_ALL, numOfAllConc, first=(i==0))
        obs2.set_forward_msg(W_O)
        obs4.set_forward_msg(W_M)
        obs6.set_forward_msg(W_P)

        # update mmlda
        for it in range(2):
            for bottom_it in range(2):
                mlda1.update()
                mlda2.update()
                mlda3.update()
            mlda_top.update()

        # 観測ノード数 + MLDA数
        module_num = 10

        n_mwz_obj, _ = mlda_p.load_model("module%03d_mlda" % (6))
        n_mwz_mot, _ = mlda_p.load_model("module%03d_mlda" % (7))
        n_mwz_pla, _ = mlda_p.load_model("module%03d_mlda" % (8))

        #相互情報量の計算
        WordsInference.main(threshold, [n_mwz_obj[1].T, n_mwz_mot[1].T, n_mwz_pla[1].T])
        #文法の学習
        bhmm.main(numOfConc, numOfAllConc)

if __name__ == '__main__':
    Learn_srk()
