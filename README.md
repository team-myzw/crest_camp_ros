# crest-camp-ros
環境準備：  
1. SigVerseの環境設定:http://www.sigverse.org/wiki/jp/?Tutorial を参考に  
2. webspeech_apiを ダウンロードする  

環境作成：  
1. SIGVerseとROSの通信を設定する  
2. Tutorialを参照しROScoreを立ち上げ、通信を起動する。  
3. HSRを環境内で操作しmapを作成する.rosrun gmapping slam_gmapping scan:=/hsrb/base_scan./hsrb/opt_command_velocityのTopicをTwist型で送信することでHSRを移動させることができる  
4. rosrun map_server map_saver でマップを保存する  

情報の保存  
1. Unitiy側でSIGverseを立ち上げ,ROSとの通信を設定する.  
2. mapフォルダ内に含まれるmap.yamlをmap_serverを用いて実行する.rosrun　map_server map_server map.yaml  
3. roslaunch でteleope_key_with_rviz.launchを起動する.rvizコンソールとamclによる自己位置推定を開始する  
4. roslaunch でrosbridge_websocket_for_audio.launchを起動する.音声認識用のweb通信用Nodeを起動する.  
5. chromeを起動し,web_speech_apiを起動する  
6. rosrun rosbag_database bag_database を起動する  
7. hsrb_joy_controllerのmarker_publisherの設定を対象とする物体に応じて設定する  
8. hsrb_joy_controller marker.launchを起動する.  
9. hsrb_joy_controller exp.launchを起動する.情報の保存を行う  
