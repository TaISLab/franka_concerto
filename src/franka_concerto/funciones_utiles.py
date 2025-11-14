#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Vector3
from scipy.spatial.transform import Rotation as R


def print_pose_named(current_pose, description="Pose"):
    """
    Imprime la información de la pose actual en la terminal.

    Args:
        pose (PoseStamped): Mensaje de ROS con la pose actual.
        description (str): Texto descriptivo para la impresión.
    """
    rospy.loginfo(f"{description}:")
    rospy.loginfo(f"  Position -> x: {current_pose.pose.position.x:.4f}, "
                  f"y: {current_pose.pose.position.y:.4f}, "
                  f"z: {current_pose.pose.position.z:.4f}")
    rospy.loginfo(f"  Orientation (Quaternion) -> x: {current_pose.pose.orientation.x:.4f}, "
                  f"y: {current_pose.pose.orientation.y:.4f}, "
                  f"z: {current_pose.pose.orientation.z:.4f}, "
                  f"w: {current_pose.pose.orientation.w:.4f}")

def calculate_quaternion_0_F(vector_target_x, vector_target_z):
    """
    Calcula la orientación de la pinza según los vectores X y Z.
    - El eje X del efector se alinea con `vector_target_x`.
    - El eje Z del efector se alinea con `vector_target_z`.
    - El eje Y se calcula como producto vectorial de z sobre x, según la regla de la mano derecha.
    Se construye la matriz de rotación con los vectores x, y, z. Se aplica la matriz de rotación.

    - param: vector_target_x: vector de sentido contrario al vect_forearm. Multiplicar por menos al llamar a la función.
    - param: vector_target_z: vector de sentido contrario al normal_vector. Multiplicar por menos al llamar a la función.

    Retorna:
        Quaternion: Orientación calculada.
    """

    vector_target_x /= np.linalg.norm(vector_target_x)
    vector_target_z /= np.linalg.norm(vector_target_z)
    vector_target_y = np.cross(vector_target_z, vector_target_x)  
    vector_target_y /= np.linalg.norm(vector_target_y)

    rotation_matrix_0_EE = np.column_stack((vector_target_x, vector_target_y, vector_target_z))
    rot_0_EE = R.from_matrix(rotation_matrix_0_EE)

    # NOTA: La configuración de pinza actual ya incluye la rotación de 45º entre flange y EE. No se debe tener en cuenta en código
    quaternion = (rot_0_EE).as_quat()

    # Construir orientación
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "fr3_link0"
    pose_msg.pose.orientation.x = quaternion[0]
    pose_msg.pose.orientation.y = quaternion[1]
    pose_msg.pose.orientation.z = quaternion[2]
    pose_msg.pose.orientation.w = quaternion[3]

    return pose_msg.pose.orientation

def calculate_gripper_position(kp_R_Wrist, normal_vector, magnitud):
    """ Calcula la posición de la pinza """
    position = Point()
    # Version para kp como point
    # position.x = kp_R_Wrist.x + normal_vector[0] * magnitud
    # position.y = kp_R_Wrist.y + normal_vector[1] * magnitud
    # position.z = kp_R_Wrist.z + normal_vector[2] * magnitud

    # Modificado porque kp es un array
    position.x = kp_R_Wrist[0] + normal_vector[0] * magnitud
    position.y = kp_R_Wrist[1] + normal_vector[1] * magnitud
    position.z = kp_R_Wrist[2] + normal_vector[2] * magnitud
    return position

def calculate_gripper_position_forearm_correction(kp_R_Wrist, normal_vector, forearm_vector, magnitud_nv, magnitud_fv):
    """ Calcula la posición de la pinza

    - param: kp_R_Wrist: Array de componentes X, Y, Z en el espacio cartesiano de la muñeca.
    - normal_vector: Vector normal al plano conformado por los KP de Hombro-Codo-Muñeca. Positivo hacia fuera del cuerpo.
    - forearm_vector: Vector entre KP Muñeca y KP Codo. Sentido positivo hacia la muñeca.
    - magnitud_nv: Distancia aplicada al vector normal para la corrección del punto.
    - magnitud_fv: Distancia aplicada al vector forearm para la corrección del punto.

    Salida:
    - Point(x, y, z): Punto 3D corregido
      
    """
    position = Point()
    # Version para kp como point
    # position.x = kp_R_Wrist.x + normal_vector[0] * magnitud
    # position.y = kp_R_Wrist.y + normal_vector[1] * magnitud
    # position.z = kp_R_Wrist.z + normal_vector[2] * magnitud

    # Modificado porque kp es un array
    position.x = kp_R_Wrist[0] + normal_vector[0] * magnitud_nv + forearm_vector[0] * magnitud_fv
    position.y = kp_R_Wrist[1] + normal_vector[1] * magnitud_nv + forearm_vector[1] * magnitud_fv
    position.z = kp_R_Wrist[2] + normal_vector[2] * magnitud_nv + forearm_vector[2] * magnitud_fv
    return position

def are_kp_valid(*keypoints):
    """
    Verifica si todos los keypoints son válidos (diferentes de 0.0).

    Args:
        keypoints (list of np.array): Lista de keypoints en 3D.

    Returns:
        bool: True si todos los keypoints son válidos.
    """
    return not any(np.all(kp == 0.0) for kp in keypoints)

def calculate_normal_vector(kp1, kp2, kp3):
    """
    Calcula el vector normal al plano definido por tres keypoints en 3D.

    Retorna:
    - normal_vector: numpy array normalizado con el vector normal al plano.
    """
    kp1, kp2, kp3 = np.array(kp1), np.array(kp2), np.array(kp3)
    v1, v2 = kp2 - kp1, kp3 - kp1
    normal_vector = np.cross(v1, v2)
    return normal_vector / np.linalg.norm(normal_vector)

def obtain_skeleton3D_callback(self, msg):
    """
    Función callback que calcula el vector normal a los KP RShoulder-RElbow-RWrist
    """
    keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])

    kp_R_Shoulder = keypointsX[6]
    kp_R_Elbow = keypointsX[8]
    kp_R_Wrist = keypointsX[10]

    if are_kp_valid(kp_R_Shoulder, kp_R_Elbow, kp_R_Wrist):
        self.normal_vector = calculate_normal_vector(kp_R_Shoulder, kp_R_Elbow, kp_R_Wrist)
        self.last_normal_vector = self.normal_vector
    else:
        rospy.logwarn("Se detectó NaN en el cálculo de la normal. No se actualiza el valor.")
        self.normal_vector = self.last_normal_vector

############################### Funciones de conversión entre np.array y Point ##################################

def np_array_to_point(np_array):
    """
    Convierte un np.array a geometry_msgs/Point.
    
    :param np_array: np.array de 3 elementos
    :return: geometry_msgs/Point con las coordenadas del np_array
    """
    if len(np_array) != 3:
        raise ValueError("El np.array debe tener 3 elementos.")
    
    point = Point()
    point.x = np_array[0]
    point.y = np_array[1]
    point.z = np_array[2]
    
    return point

def np_array_to_vector3(np_array):
    """
    Convierte un np.array a geometry_msgs/Vector3.
    
    :param np_array: np.array de 3 elementos
    :return: geometry_msgs/Vector3 con las coordenadas del np_array
    """
    if len(np_array) != 3:
        raise ValueError("El np.array debe tener 3 elementos.")
    
    vector = Vector3()
    vector.x = np_array[0]
    vector.y = np_array[1]
    vector.z = np_array[2]
    
    return vector

def point_to_np_array(point):
    """
    Convierte geometry_msgs/Point a un np.array.
    
    :param point: geometry_msgs/Point
    :return: np.array con las coordenadas del Point
    """
    return np.array([point.x, point.y, point.z])

def vector3_to_np_array(vector):
    """
    Convierte geometry_msgs/Vector3 a un np.array.
    
    :param vector: geometry_msgs/Vector3
    :return: np.array con las coordenadas del Vector3
    """
    return np.array([vector.x, vector.y, vector.z])







