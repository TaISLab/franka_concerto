#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from std_msgs.msg import Int32, Bool, Float32  # Importa el tipo de mensaje Int32
from scipy.spatial.transform import Rotation as R

from franka_buttons.msg import FrankaButtons
from franka_msgs.msg import FrankaState

# Este nodo se subscribe a distintos topics, los analiza y publica el estado deseado del sistema.
# FRANKA_BUTTONS: para detectar valores en X e Y es necesario que el Franka Desk esté en pilot mode -> tablet/pc y no en franka hand

class hriStatePublisher:
    def __init__(self):
        """
        Initialize the ROS node, set up subscribers.
        """
        rospy.init_node('hri_states_publisher', anonymous=False)

        # Publishers
        self.hri_state_pub = rospy.Publisher('/hri_state_desired', Int32, queue_size=10)
        self.gripper_state_d_pub = rospy.Publisher('/gripper_state_desired', Int32, queue_size=10)

        # Subscripción a la info de los botones del Franka FR3.
        rospy.Subscriber('/franka_buttons', FrankaButtons, self.franka_buttons_callback)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.obtain_franka_info_callback)

        # Estado HRI. se inicia en -1 para detectar el primer cambio
        self.hri_state_desired = -1
        self.operator_hri_state_desired = -1 # Estado que desea el operador con los botones, puede no ser posible
        
        self.gripper_state_d = -1  

        # Info del Franka
        self.franka_state = FrankaState.robot_mode # Estado del manipulador
        self.cartesian_contact = FrankaState.cartesian_contact # Contactos cartesianos. Se define cuando ocurre un contacto dentro de franka_control.yaml

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
        self.prev_x = 0.0
        self.prev_y = 0.0
    
    def obtain_franka_info_callback(self, msg):
        """
        Actualiza la información obtenida de franka_states
        """
        rospy.loginfo("Actualizando estado del robot")
        self.franka_state = msg.robot_mode
        self.cartesian_contact = msg.cartesian_contact


    def franka_buttons_callback(self, msg):
        """
        Callback para actualizar los valores de los botones del Franka y resetearlos si cambian de 0 a 1.

        :param msg: Mensaje de tipo FrankaButtons con el estado de los botones.
        """
        # Detectar cambios de 0 -> 1 y resetear a 0 inmediatamente

        # Botón CROSS
        if msg.cross and not self.prev_cross:  # Detecta flanco positivo. De 0 a 1
            rospy.loginfo("Botón Cross detectado")
            self.operator_hri_state_desired = 0

        # Botón CHECK
        if msg.check and not self.prev_check:  # Detecta flanco positivo
            rospy.loginfo("Botón Check detectado")
            self.operator_hri_state_desired = 1

        # Botón CIRCLE
        if msg.circle and not self.prev_circle:  # Detecta flanco positivo
            rospy.loginfo("Botón Circle detectado")
            self.operator_hri_state_desired = 2


        # Botón X: Control de la pinza (Cierra si X=1, Abre si X=-1)
        if msg.x != self.prev_x:  # Detecta cambio de estado en el joystick
            if msg.x == 1:
                rospy.loginfo("Botón x = 1. Cerrando pinza")
                self.gripper_state_d = 1
            elif msg.x == -1:
                rospy.loginfo("Botón x = 1. Abriendo pinza")
                self.gripper_state_d = 0

            self.prev_x = msg.x  # Actualizar el estado previo del joystick
            self.gripper_state_d_pub.publish(self.gripper_state_d)

        # Actualizar estados previos de los botones (para detectar flanco positivo en la siguiente iteración)
        self.prev_cross = msg.cross
        self.prev_check = msg.check
        self.prev_circle = msg.circle

        # Actualizar joystick
        self.x = msg.x
        self.y = msg.y


    def hri_states_publisher_callback(self):
        '''
        Publica el estado deseado
        '''

        if self.operator_hri_state_desired == 1:
            rospy.logdebug("Se publica el estado 1. Aproximación")
            self.hri_state_desired = 1
            self.hri_state_pub.publish(self.hri_state_desired)

        elif self.operator_hri_state_desired == 2:
            rospy.logdebug("Se publica el estado 2. Agarre")
            self.hri_state_desired = 2
            self.hri_state_pub.publish(self.hri_state_desired)

        else:
            rospy.logdebug("Se publica el estado 0. Reposo")
            self.hri_state_desired = 0
            self.hri_state_pub.publish(self.hri_state_desired)

        self.last_hri_state_desired = self.hri_state_desired


if __name__ == '__main__':
    try:
        # Crear instancia del publicador
        hri_state_publisher = hriStatePublisher()

        def shutdown_callback():
            """
            Handles the shutdown of the ROS node, ensuring clean closure of resources.
            """
            rospy.loginfo("Shutting down hri_state_publisher node...")
        
        # Registrar el callback de apagado
        rospy.on_shutdown(shutdown_callback)

        rate = rospy.Rate(10) # tasa de ejecución del bucle. 10 Hz. 10 veces por segundo

        while not rospy.is_shutdown():
            hri_state_publisher.hri_states_publisher_callback()
            rate.sleep()  # Esperar para mantener la frecuencia

    except rospy.ROSInterruptException:
        pass  # Manejo de cierre seguro de ROS
