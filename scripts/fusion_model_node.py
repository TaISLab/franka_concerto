#!/usr/bin/env python3


# Este nodo se va a encargar de substituir la muñeca por el EE. Va a calcular el vector normal, el vector forearm, el vector upperarm y lo va a a publicar en un topic /fusion_model
# Una posible idea del mensaje puede ser:
# q1, q2, q3, q4, normal_vector upperarm_vector forearm_vector upperarm_length forearm_length kp_R_Shoulder kp_R_Elbow kp_R_Wrist_gripped

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Vector3
from franka_concerto.funciones_utiles import are_kp_valid, calculate_normal_vector, np_array_to_point, np_array_to_vector3, point_to_np_array, vector3_to_np_array
from franka_concerto.visualization_utils import VectorVisualizer
from skeleton_3d.msg import Skeleton3D
from franka_concerto.msg import FusionModel

class FusionModelPublisher:
    def __init__(self):
        rospy.init_node("fusion_model_publisher", log_level=rospy.DEBUG)

        # Publicadores
        self.fusion_model_publisher = rospy.Publisher("/fusion_model", FusionModel, queue_size=1)

        # Subscriptores
        rospy.Subscriber('/robot_vel_ctrl/pose', PoseStamped, self.current_pose_callback, queue_size=1) # Subscripción a la pose actual
        rospy.Subscriber('/skeleton_3D', Skeleton3D, self.kp_callback, queue_size=1)

        self.kp_R_Shoulder = np.array([0.0, 0.0, 0.0])
        self.kp_R_Elbow = np.array([0.0, 0.0, 0.0])
        self.position_EE = np.array([0.0, 0.0, 0.0])

    def current_pose_callback(self, msg):
        """ Callback que obtiene la pose actual del robot desde el nuevo controlador """
        self.position_EE = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])


    def kp_callback(self, msg):
        """
        Callback que obtiene los puntos del sistema de visión
        
        Guide of Kp:
        #  5 left_shoulder
        #  6 right_shoulder
        #  7 left_elbow
        #  8 right_elbow
        #  9 left_wrist
        # 10 right_wrist
        # 11 left_hip
        # 12 right_hip

        """
        keypoint = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])
        
        # Asignacion de Kp
        self.kp_R_Shoulder = keypoint[6]
        self.kp_R_Elbow = keypoint[8]
        self.kp_R_Wrist = keypoint[10]

        self.kp_L_Shoulder = keypoint[5] 

    def fusion_model_calculator(self):
        """
        Función que calcula y publica el modelo fusionado
        """

        fusion_model = FusionModel()

        # Pasar de Array a Point
        fusion_model.kp_R_shoulder= np_array_to_point(self.kp_R_Shoulder)
        fusion_model.kp_R_elbow= np_array_to_point(self.kp_R_Elbow)
        fusion_model.kp_R_wrist_gripped= np_array_to_point(self.position_EE)
        
        
        
        if are_kp_valid(self.kp_R_Shoulder, self.kp_R_Elbow, self.position_EE):
            # Calcular normal_vector
            normal_vector = calculate_normal_vector(self.kp_R_Shoulder, self.kp_R_Elbow, self.position_EE)
            fusion_model.normal_vector = np_array_to_vector3(normal_vector)
             
            # Calcular upperarm_vector
            upperarm_vector = self.kp_R_Elbow - self.kp_R_Shoulder
            fusion_model.upperarm_vector = np_array_to_vector3(upperarm_vector)
            
            # Calcular forearm_vector
            forearm_vector = self.position_EE - self.kp_R_Elbow
            fusion_model.forearm_vector = np_array_to_vector3(forearm_vector)
            
            # Calcular longitudes
            fusion_model.upperarm_length = np.linalg.norm(upperarm_vector)
            fusion_model.forearm_length = np.linalg.norm(forearm_vector)
            
            if are_kp_valid(self.kp_R_Wrist):
                fusion_model.kp_R_wrist_non_gripped= np_array_to_point(self.kp_R_Wrist)

            if are_kp_valid(self.kp_L_Shoulder):
                vector_q1 = self.kp_R_Shoulder - self.kp_L_Shoulder
                vector_q1 /= np.linalg.norm(vector_q1)
                fusion_model.vector_q1= np_array_to_vector3(vector_q1)

            # Publicar mensaje
            rospy.logdebug("Fusion Model: Mensaje publicado en topic /fusion_model")

            ## Rodri del futuro: Considera publicar cuando se verifique que se está agarrando, por ejemplo: la fsm publica is_gripped=true, y este nodo lo lee y publica
            self.fusion_model_publisher.publish(fusion_model)

        
        else:
            rospy.logdebug("Fusion Model: Mensaje no publicado en topic /fusion_model. Condición are_kp_valid evaluada como falsa")
            rospy.sleep(1)

if __name__ == '__main__':
    
    # Create an instance of the ROSInterface and start listening for messages
    fusion_model_node = FusionModelPublisher()
    rospy.loginfo("Fusion Model: Nodo iniciado correctamente")
    
    rate_hz = 30 # En Hz
    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        # Bucle de ejecución
        fusion_model_node.fusion_model_calculator()
        rate.sleep()
    
    # Aqui solo llega cuando se solicita cerrar el nodo
    rospy.loginfo("Fusion model: cerrando nodo de manera controlada")

    

    