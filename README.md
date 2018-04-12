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

1. 以下のコマンドを実行し，コントローラによりロボットを操作しmapを作成する.  
 `rosrun gmapping slam_gmapping scan:=/hsrb/base_scan`  
 `roslaunch hsrb_joy_controller exp.launch`  

2. 以下のコマンドでmapを保存する  
`roscd crest_camp_ros`  
`mkdir map`  
`cd map`  
`rosrun map_server map_saver`

### SIGVerseでの情報取得:  
1. WindowsPCでSIGverseを立ち上げ,ROSとの通信を設定する.  
2. mapフォルダ内に含まれるmap.yamlをmap_serverを用いて実行する.rosrun　map_server map_server map.yaml  
3. `roslaunch crest_camp_ros main.launch`
4. web_speech_api[ coming soon ]を起動する  
5. `roscd rosbag_database/src`  
6. `rosrun rosbag_database bag_database.py`  
7. hsrb_joy_controllerのmarker_publisherの設定を対象とする物体に応じて設定する  
8. `roslaunch hsrb_joy_controller marker.launch`
9. `roslaunch hsrb_joy_controller exp.launch`
10. ロボットを操作しながら，ロボットの行動に対応した発話を行う．

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
* [crest-camp-unity](https://github.com/team-myzw/crest-camp-unity)の環境をwindows・ubuntu共に実行
* [gp_hsmm_action](https://github.com/team-myzw/gp_hsmm_action)の学習動作実行方法に従い，動作生成用プログラムを準備
* [spco_formationのnavigation](https://github.com/team-myzw/spco_formation#navigation)に従い，navigationの準備
* `rosrun crest_camp_ros sentence2task.py`
* `/crest_camp_ros/order`に命令文を送る
