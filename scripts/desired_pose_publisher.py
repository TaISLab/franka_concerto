#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Quaternion

class DesiredPosePublisher:
    def __init__(self):
        rospy.init_node("desired_pose_publisher")
        self.desired_pose_publisher = rospy.Publisher("/desired_pose", PoseStamped, queue_size=10)

if __name__ == '__main__':
    try:
        rospy.loginfo("Nodo iniciado correctamente")
        pose_publisher = DesiredPosePublisher()

        # Posicion
        target_pose = PoseStamped()
        target_pose.pose.position.x = 0.5
        target_pose.pose.position.y = 0.0
        target_pose.pose.position.z = 0.5

        # Orientacion. Definida como ángulos Euler XYZ. 
        euler_angles = [3.1415, 0, 0]  # (x, y, z) en radianes. La rotx(pi) para efector final hacia abajo

        quat = R.from_euler('xyz', euler_angles).as_quat() # Convertir a cuaternión

        q_norm = R.from_quat(quat).as_quat()  # scipy automáticamente normaliza
        
        # Convertir a quaternion de geometry_msg
        q_ros = Quaternion()
        q_ros.x = q_norm[0]
        q_ros.y = q_norm[1]
        q_ros.z = q_norm[2]
        q_ros.w = q_norm[3]

        target_pose.pose.orientation = q_ros

        rospy.sleep(2)  # Esperar que todo esté listo antes de enviar

        # Bucle de control
        rate = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
                target_pose.header.stamp = rospy.Time.now()
                target_pose.header.frame_id = "fr3_link0"
                pose_publisher.desired_pose_publisher.publish(target_pose)
                rate.sleep()
            
        # Salida controlada del nodo
        def shutdown_callback():
            rospy.loginfo("Shutting down desired_pose node...")
        rospy.on_shutdown(shutdown_callback)

    except rospy.ROSInterruptException:
        pass

