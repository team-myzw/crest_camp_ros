# crest-camp-ros
遠隔操作により取得したデータを用いて行動と言語を学習．  
学習したモデルを用いることで，言語命令からタスクを実行する．

## Dependencies  
* [sigverse_ros_package](https://github.com/SIGVerse/sigverse_ros_package)
* [SigVerse ver.3](http://www.sigverse.org/wiki/jp/?Tutorial)
* [crest-camp-unity](https://github.com/team-myzw/crest-camp-unity)
* [webspeech_api]() [coming soon]
* [spco_formation](https://github.com/team-myzw/spco_formation)
* [gp_hsmm_action](https://github.com/team-myzw/gp_hsmm_action)
* [mmlda_bhmm](https://github.com/team-myzw/mmlda_bhmm)
* [hsrb_joy_controller](https://github.com/team-myzw/hsrb_joy_controller)
* [rosbag_database](https://github.com/team-myzw/rosbag_database)

## SIGVerseの起動方法
* Windows PC: Unityで crest-camp-unityを実行
* Ubuntu PC: `roslaunch crest_camp_ros main.launch`

## 遠隔操作によるデータ取得

### 地図作成：  

1. コントローラによりロボットを操作しmapを作成  
 `rosrun gmapping slam_gmapping scan:=/hsrb/base_scan`  
 `roslaunch hsrb_joy_controller exp.launch`  

2. mapの保存  
`roscd crest_camp_ros`  
`mkdir map`  
`cd map`  
`rosrun map_server map_saver`

### SIGVerseでの情報取得:  
1. hsrb_joy_controllerのmarker_publisherの設定を対象とする物体に応じて設定する
1. WindowsPCでSIGverseを立ち上げる
1. `roslaunch crest_camp_ros get_info.launch`
1. web_speech_api[ coming soon ]を起動する  
1. Unityでゲームをスタート
1. `roscd rosbag_database/src`  
1. `rosrun rosbag_database bag_database.py`  
1. ロボットを操作しながら，ロボットの行動に対応した発話を行う．

## モデルの学習
### 場所概念の学習
[spco_formation](https://github.com/team-myzw/spco_formation)を参考  
追記予定

### 動作の学習
[gp_hsmm_action](https://github.com/team-myzw/gp_hsmm_action)を参考  
追記予定

### 統合概念・言語の学習
[mmlda_bhmm](https://github.com/team-myzw/mmlda_bhmm)を参考  
追記予定

## 言語命令によるタスク実行  
1. hsrb_joy_controllerのmarker_publisherの設定を対象とする物体に応じて設定する
1. WindowsPCでSIGverseを立ち上げる
1. `roslaunch crest_camp_ros get_info.launch`
1. web_speech_api[ coming soon ]を起動する  
1. Unityでゲームをスタート
1. [gp_hsmm_action](https://github.com/team-myzw/gp_hsmm_action)の学習動作実行方法に従い，動作生成用プログラムを準備
1. `roslaunch crest_camp_ros task_execution.launch`
1. `roscd em_spco_formation/src/`
1. `python em_name2place.py <<TRIALNAME>>`
1. web_speech_apiによる音声認識で命令文を送る [coming soon]
