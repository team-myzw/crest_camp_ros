# -*- coding: utf-8 -*-
#from __future__ import unicode_literals
import os
import shutil
import wave
import threading
import numpy as np
import glob
import time
import sys
from codecs import open

#import Mecab #中友さん自作Mecab用パッケージ
import MeCab # 普通のMeCab

#import rospy

class WordClient(object):
    """
    触覚/聴覚のモダリティと異なり，これは actionlib を実行するわけではない．

    """
    def __init__(self):
        """
        """
        # Mecab
        self.m = MeCab.Tagger("-Owakati")

        # attributes
        self.is_accept = False  # 情報取得中のみ True になる変数

        # data
        self._wav_files = list()
        self._wav_dir = None
        self._num_sentence = 0
        self._utterd_words = list()
        self._lock = threading.RLock()

        self._codebook = list()


    def wait_for_service(self, service):
        try:
            service.wait_for_service(3)
        except (rospy.ROSException):
            name = service.resolved_name
            rospy.logwarn("wait for {name}".format(name=name))
            service.wait_for_service()

    def update(self, wav_dir):
        """
        (1)wavfファイルを保存するディレクトリを変更し，
        (2)今まで保存していたwav ファイルのパスを返すメソッド

        Parameters
        ----------
        wav_dir: str
            wav を保存するディレクトリパス

        Returns
        -------
        wav_files: list of str
            update メソッドが呼ばれるまでに保存された wav fileのパスが格納されたリスト
        """
        try:
            os.makedirs(wav_dir)
        except:
            pass

        with self._lock:
            # 保存すべきディレクトリを更新する
            self._wav_dir = wav_dir
            # 今まで tmp/ に保存した wav を _wav_dir/ 以下にコピー
            wav_files = list()
            self._num_sentence=0
            # for wav in self._wav_files:
            #     file_name = os.path.basename(wav)
            #     dst_wav = os.path.join(self._wav_dir, file_name)
            #     shutil.move(wav, dst_wav)
            #     wav_files.append(dst_wav)
            # self._wav_files = list()
            return wav_files

    def setAccept(self):
        with self._lock:
            self.is_accept = True

    def setReject(self):
        with self._lock:
            self.is_accept = False

    def audio_data_cb(self, audio_data):
        """
        音声の wave データが送信されてきたときに受け取るコールバック関数

        Parameters
        -----------
        audio_data: AudioData
        """
        with self._lock:
            # is_accept がFalseの間は情報取得を行わない
            if not self.is_accept:
                return
            # 保存する wav の名前を決める
            num_saved_wav = len(self._wav_files)
            wav_file_name = os.path.join(self._wav_dir, "{0:03d}.wav".format(num_saved_wav))
            try:
                wav = wave.open(wav_file_name, "w")
            except (Exception) as e:
                rospy.logerr("{e}".format(e=e))
                return
            else:
                # wav の設定
                wav.setframerate(audio_data.sampling_rate)
                wav.setsampwidth(audio_data.sample_width)
                wav.setnchannels(audio_data.channels)
                # wav を書き出す
                data = np.array(audio_data.data, dtype=np.short)
                wav.writeframes(data.tostring())
                # 保存した履歴に足す
                self._wav_files.append(os.path.abspath(wav_file_name))
            finally:
                wav.close()

    def sentence_data_cb(self, sentence_data):
        u"""
        音声認識されたsentenceデータが送信されたときに結果をファイルに書き起こすコールバック関数

        params
        -----
        sentence_data:SpeechRecognitionCandidates
        """
        with self._lock:
            # is_accept がFalseの間は情報取得を行わない
            if not self.is_accept:
                return
            # 保存する sentence の名前を決める
            sentence_file_name = os.path.join(self._wav_dir, "sentence_{0:03d}.txt".format(self._num_sentence))
            f = open(sentence_file_name, "w")
            f.write(sentence_data)
	    # f.write(sentence_data.sentences[0])
            f.close()
            self._num_sentence += 1

    def set_DragonSpeech_dirs(self, observe_dir, output_dir):
        u"""
        ドラゴンスピーチによるwavファイルの音声認識に関わるディレクトリを設定するメソッド

        params
        -----
        observe_dir : str
            ドラゴンスピーチの自動エージェントがwavファイルの保存を監視しているディレクトリのパス
        output_dir　： str
            ドラゴンスピーチがwavファイルの認識結果を保存するディレクトリのパス
        """
        self.observe_dir = observe_dir
        self.output_dir = output_dir

    def recognize_with_DragonSpeech(self, src_dir, save_dir):
        u"""
        取得した発話のwavファイルをDoragonSpeechで音声認識させるメソッド
        [手順]
        1.DoragonSpeechが監視するディレクトリに移動させ，音声認識
        2.音声認識結果を保存ディレクトリに移動

        params
        -----
        src_dir : str
            取得した発話のwavファイルを保存しているディレクトリのパス
        save_dir : str
            音声認識結果を保存するディレクトリのパス
        """
        # 1. src_dir内のwavファイルをDoragonSpeechが監視するディレクトリに移動
        wav_files = glob.glob(os.path.join(src_dir, "*.wav"))
        for w in wav_files:
            shutil.move(w, self.observe_dir)

        # 2.音声認識が終わるまで待機
        while len(glob.glob(os.path.join(self.observe_dir, "*.wav"))) != 0:
            print ("recognize...")
            time.sleep(1.0)

        # 3.認識結果およびwavファイルを指定した保存ディレクトリに移動
        wav_files = glob.glob(os.path.join(self.output_dir, "*.wav"))
        for w in wav_files:
            shutil.move(w, save_dir)

        result_files = glob.glob(os.path.join(self.output_dir, "*.txt"))
        for r in result_files:
            shutil.move(r, save_dir)

    def move_wave_and_sentence(self, tmp_dir, save_dir):
        u"""
        取得した発話のwavファイルをと音声認識結果を保存ディレクトリに移動

        params
        -----
        src_dir : str
            取得した発話のwavファイルを保存しているディレクトリのパス
        save_dir : str
            音声認識結果を保存するディレクトリのパス
        """
        # 認識結果およびwavファイルを指定した保存ディレクトリに移動
        files = glob.glob(os.path.join(tmp_dir, "*"))
        for w in files:
            shutil.move(w, save_dir)

    def _split_sentence(self, sentence_file, save_file):
        u"""
        MeCabによる構文解析を行って，sentenceをwordに分割するメソッド

        Params
        -----
        sentence_file : str
            対象の文が記録されたファイルのパス
        save_file : str
            単語分割結果を保存するファイルのパス
        """
        #sentences = Mecab.ParseFile(sentence_file)
        sentences = None
        with open(sentence_file, "r", encoding="utf-8") as f:
            sen = f.read()
            sen = sen.encode("utf-8") # unicode to str
            sentences = self.m.parse(sen)
            print sentences
        sentences = sentences.decode("utf-8") # str to unicode
        with open(save_file, "w", encoding="utf-8") as g:
            # for words in sentences:
            #     print words
            #     for w in words:
            #         print w
            #
            #         g.write(w.surfaceForm+"\t")
            #         #g.write(w+"\t")
            g.write(sentences)

    def split_sentences(self, target_dir):
        u"""
        対象ディレクトリ内の"sentence_*.txt"に対して、それぞれ"\t"区切りのファイルを作成するメソッド
        """
        srces = glob.glob(os.path.join(target_dir, "sentence_*.txt"))
        for n,s in enumerate(srces):
            self._split_sentence(s, os.path.join(target_dir, "split_sentence_%03d.txt"%n))

    def sumalize_utter(self, target_dir, dstfile_name="sentences.txt"):
        u"""
        対象ディレクトリ（target_dir）に含まれる　"split_sentence_*.txt" を1つのファイル["sentences.txt"]にまとめるメソッド

        Params
        -----
        target_dir : str
            対象ファイルを保存しているディレクトリのパス
        dstfile_name : str
            まとめた結果を保存するファイル名
        """
        sentence_files = glob.glob(os.path.join(target_dir, "split_sentence_*.txt"))
        with open(os.path.join(target_dir, dstfile_name), "w", encoding="utf-8") as g:
            for s in sentence_files:
                sentence = open(s, "r", encoding="utf-8").readline()
                print sentence
                g.write(sentence+"\n")

    def words2histogram(self, srcfile, dstfile, num_dim=67):
        """
        単語の発生頻度で表現したヒストグラムに変換するメソッド
        ※　事前に"word_codebook.txt"を作成し，同ディレクトリに保存しておくこと

        params
        -----
        srcfile : str
            単語区切りの発話文章のファイル
        num_dim : int
            ヒストグラムの次元数
        """
        # 単語とインデックスの対応を表すdictionaryの作成
        codebook = self._codebook
        word2idx_dic = dict()
        for i,w in enumerate(codebook):
            word2idx_dic.update({w:i})

        #データをヒストグラムに変換
        histogram = np.zeros(num_dim)
        sentences = open(srcfile, "r",).readlines()
        for s in sentences:
            words = s.split()
            for w in words:
                idx = word2idx_dic[str(w)]
                histogram[idx] = histogram[idx] + 1

        filename = dstfile
        np.savetxt(filename, [histogram], fmt="%d", delimiter=" ")



    def update_word_codebook(self, sentences_file_path):
        """
        単語モダリティのコードブックを更新をするメソッド．

        Params
        -----
        sentences_file_path : str
            単語を追加する文章データのファイルパス
        """

        # 追加データの読み込みとcodebookの更新
        with open(sentences_file_path, "r") as f:
            sentences = f.readlines()
            for s in sentences:
                words = s.split()
                for w in words:
                    if not w in self._codebook:
                        self._codebook.append(w)

    def dump_codebook(self,file_path):
        np.savetxt(file_path,np.array(self._codebook),fmt="%s")

    def load_codebook(self,file_path):
        self._codebook = np.loadtxt(file_path,dtype=str).tolist()


def sample():

    client = WordClient()
    for i in range(5):
        client.setAccept()
        # target_dir = "data/{0:04d}/word".format(i)
        tmp = os.path.join("data", "{0:04d}".format(i))
        target_dir = os.path.join(tmp, "word")
        print target_dir
        for j in range(10):
            print j
            time.sleep(1)
        client.setReject()
        #client.recognize_with_DragonSpeech("tmp", target_dir)
        client.move_wave_and_sentence("tmp", target_dir)
        client.split_sentences(target_dir)
        client.sumalize_utter(target_dir)
        client.update_word_codebook(i, 1, i is 0)
        print os.path.join(target_dir,"sentences.txt")
        client.words2histogram(os.path.join(target_dir,"sentences.txt"), os.path.join(target_dir, "hist.txt"))
    print "sample end"

def main():
    #start node
    rospy.init_node("word_client_for_Dragon")
    sample()
    rospy.spin()

if __name__ == "__main__":
    main()
