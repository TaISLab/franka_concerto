#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from std_msgs.msg import Int32, Bool, Float32  # Importa el tipo de mensaje Int32
from scipy.spatial.transform import Rotation as R

from franka_buttons.msg import FrankaButtons


class hriStatePublisher:
    def __init__(self):
        """
        Initialize the ROS node, set up subscribers.
        """
        rospy.init_node('eq_pose_publisher', anonymous=False)

        # Publisher para enviar el equilibrium_pose
        self.hri_state_pub = rospy.Publisher('/hri_state', Int32, queue_size=10)

        # Subscripción a la info de los botones del Franka FR3.
        rospy.Subscriber('/franka_buttons', FrankaButtons, self.franka_buttons_callback)

        # Estado HRI. se inicia en -1 para detectar el primer cambio
        self.hri_state = -1  

        # Estado actual de los botones
        self.check = False
        self.circle = False
        self.cross = False
        self.x = 0.0
        self.y = 0.0

        # Estado anterior de los botones
        self.prev_check = False
        self.prev_circle = False
        self.prev_cross = False

    def franka_buttons_callback(self, msg):
        """
        Callback para actualizar los valores de los botones del Franka y resetearlos si cambian de 0 a 1.

        :param msg: Mensaje de tipo FrankaButtons con el estado de los botones.
        """
        # Detectar cambios de 0 -> 1 y resetear a 0 inmediatamente
        if not self.prev_cross and msg.cross:
            rospy.loginfo("Botón Cross presionado")
            self.hri_state = 0
            self.hri_state_pub.publish(self.hri_state)

            self.cross = False
        else:
            self.cross = msg.cross
        
        if not self.prev_check and msg.check:
            rospy.loginfo("Botón Check presionado")
            self.hri_state = 1
            self.hri_state_pub.publish(self.hri_state)

            self.check = False
            
        else:
            self.check = msg.check

        if not self.prev_circle and msg.circle:
            rospy.loginfo("Botón Circle presionado")
            self.hri_state = 2
            self.hri_state_pub.publish(self.hri_state)
            self.circle = False
        else:
            self.circle = msg.circle

        # Guardar el estado anterior de los botones
        self.prev_check = msg.check
        self.prev_circle = msg.circle
        self.prev_cross = msg.cross

        # Actualizar joystick normalmente
        self.x = msg.x
        self.y = msg.y


if __name__ == '__main__':
    try:
        # Se crea una instancia de la clase y comienza a leer mensajes
        hri_state_publisher = hriStatePublisher()

        def shutdown_callback():
            """
            Handles the shutdown of the ROS node, ensuring clean closure of resources.
            """
            rospy.loginfo("Shutting down hri_state_publisher node...")

        rospy.on_shutdown(shutdown_callback)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        pass
