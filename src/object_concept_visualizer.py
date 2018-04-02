#!/usr/bin/env python
import rospy

import math
import tf2_ros

# import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math


OBJECTS=["apple_01","banana_01","doll_bear_01","doll_dog_01","doll_rabbit_01",
         "cannedjuice_350ml_01","petbottle_2l_empty_c01_01","petbottle_500ml_full_c01_01"]

CCATEGORY_NAME = ["PET", "Fruits", "Stuffed toy"]

topic = '/crest_camp/message/object_concept'
publisher = rospy.Publisher(topic, Marker)

def marker_set(frame_id, pose, text):
    marker = Marker()
    marker.header.frame_id = "/neck"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.text = text
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = pose.x
    marker.pose.position.y = pose.y
    marker.pose.position.z = pose.z
    publisher.publish(marker)
    # return marker

if __name__ == '__main__':
    rospy.init_node('object_concept_visualizer')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    objects = OBJECTS
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        for object_name in objects:
            try:
                trans = tfBuffer.lookup_transform("odom", object_name, rospy.Time())
                print trans
                # conc = id2concept(object_name)
                # word = concept2word(conc)
                word = "ika"
                marker_set(object_name, trans.transform.translation, word)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue


        rate.sleep()
