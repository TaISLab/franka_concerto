#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from sensor_msgs import point_cloud2

def cb(msg):
    points = [(p.x, p.y, p.z) for p in msg.points]
    header = msg.header
    pc2 = point_cloud2.create_cloud_xyz32(header, points)
    pub.publish(pc2)

rospy.init_node("pc1_to_pc2")
pub = rospy.Publisher("/JSK_NODELET_jsk_pcl_ros_euclidean_clustering/input", PointCloud2, queue_size=1)
sub = rospy.Subscriber("/natnet_ros/pointcloud", PointCloud, cb)
rospy.spin()