#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from human_articular_space.msg import KP_URDF
from std_msgs.msg import Int32  # Importa el tipo de mensaje Int32
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP
from scipy.spatial.transform import Rotation as R
from franka_msgs.msg import FrankaState


def print_current_pose(current_pose, description="Current Pose"):
    """
    Imprime la información de la pose actual en la terminal.

    Args:
        current_pose (PoseStamped): Mensaje de ROS con la pose actual.
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

def smooth_interpolation(current_pose, target_pose, max_step=0.01):
    """
    Interpola suavemente entre la pose actual y la pose objetivo.

    Args:
        current_pose (PoseStamped): Pose actual del efector final.
        target_pose (PoseStamped): Pose objetivo del efector final.
        max_step (float): Máximo cambio de posición permitido por ciclo.

    Returns:
        PoseStamped: Nueva pose suavizada.
    """
    new_pose = PoseStamped()
    new_pose.header.stamp = rospy.Time.now()
    new_pose.header.frame_id = target_pose.header.frame_id
    
    # Interpolación lineal con un límite de velocidad
    for i, axis in enumerate(["x", "y", "z"]):
        current = getattr(current_pose.pose.position, axis)
        target = getattr(target_pose.pose.position, axis)
        delta = np.clip(target - current, -max_step, max_step)
        setattr(new_pose.pose.position, axis, current + delta)

    # Mantener la orientación sin cambios
    new_pose.pose.orientation = target_pose.pose.orientation

    return new_pose

def calculate_quaternion_0_F(vector_target_y, vector_target_z):
    """
    VERIFICADO. RCO. 10-02-25
    Calcula la orientación de la pinza según dos vectores:
    - El eje Y del efector se alinea con `vector_target_y`.
    - El eje Z del efector se alinea con `vector_target_z`.
    El tercer vector se calcula con producto vectorial de los dos primeros.
    Posteriormente se calcula la matriz de rotación entre 0 y EE.
    Luego, se aplica la rotación sobre ejes locales Rot_EE_F para aplicar la rotación entre el EE y la pinza. 45 grados en el eje Z.
    
    Entrada:
        vector_target_y (np.array): Vector de referencia 3D para el eje Y del EE. No es necesario normalizar fuera.
        vector_target_z (np.array): Vector de referencia 3D para el eje Z del EE. No es necesario normalizar fuera.

    Salida:
        geometry_msgs/PoseStamped: Mensaje de pose con la orientación calculada.
    """
    # Normalizar vectores
    vector_target_y = vector_target_y / np.linalg.norm(vector_target_y)
    vector_target_z = vector_target_z / np.linalg.norm(vector_target_z)
    
    # Calcular vector x como producto cruz
    vector_target_x = np.cross(vector_target_y, vector_target_z) # VERIFICADO. PRODUCTO CRUZ PARA SACAR EL OTRO VECTOR
    vector_target_x /= np.linalg.norm(vector_target_x) # Normalizar vector x

    # Matriz de rotación deseada entre 0 y EE
    rotation_matrix_0_EE = np.column_stack((vector_target_x, vector_target_y, vector_target_z))

    # Convertir la matriz de rotación a un objeto de rotación de scipy
    rot_0_EE = R.from_matrix(rotation_matrix_0_EE)
    
    # Alinear ejes. la herramienta está girada -45º en Z respecto al EE. F es la nomenclatura de la herramienta segun Franka
    rot_EE_F = R.from_rotvec(np.radians(-45) * np.array([0, 0, 1]))  # Rotación de -45° en z local
    quaternion = (rot_0_EE * rot_EE_F).as_quat() # Aplicar rotación

    # Construir el mensaje de ROS PoseStamped
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "fr3_link0"
    pose_msg.pose.orientation.x = quaternion[0]
    pose_msg.pose.orientation.y = quaternion[1]
    pose_msg.pose.orientation.z = quaternion[2]
    pose_msg.pose.orientation.w = quaternion[3]

    return pose_msg.pose.orientation

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''

    return not any(np.all(kp == 0.0) for kp in keypoints)

def calculate_normal_vector(kp1, kp2, kp3):
    """
    Calcula el vector normal al plano definido por tres keypoints en 3D.

    Parámetros:
    - kp1, kp2, kp3: numpy arrays o listas con coordenadas (x, y, z) de los puntos.

    Retorna:
    - normal_vector: numpy array normalizado con el vector normal al plano.
    """
    # Convertir los keypoints en numpy arrays
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    kp3 = np.array(kp3)

    # Crear dos vectores en el plano
    v1 = kp2 - kp1
    v2 = kp3 - kp1

    # Calcular el producto cruzado (vector normal al plano)
    normal_vector = np.cross(v1, v2)

    # Normalizar el vector (para que tenga magnitud 1)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector

class EquilibriumPosePublisher:
    def __init__(self):
        """
        Initialize the ROS node, set up subscribers.
        """
        rospy.init_node('eq_pose_publisher', anonymous=False)

        # Publisher para enviar el equilibrium_pose
        self.pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)

        # Subscriber para recibir datos de KP_URDF
        rospy.Subscriber('/kp_URDF', KP_URDF, self.kp_callback)
        rospy.Subscriber('/hri_state', Int32, self.obtain_hri_state_callback)
        rospy.Subscriber('/skeleton_3D', Skeleton3D, self.obtain_skeleton3D_callback)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.obtain_current_pose_callback)

        # Inicializar pose con valores por defecto
        self.desired_pose = PoseStamped()
        self.desired_pose.header.frame_id = "fr3_link0"
        self.desired_pose.pose.position.x = 0.25
        self.desired_pose.pose.position.y = 0.0
        self.desired_pose.pose.position.z = 0.5
        self.desired_pose.pose.orientation.x = 0.92
        self.desired_pose.pose.orientation.y = -0.37
        self.desired_pose.pose.orientation.z = 0.0
        self.desired_pose.pose.orientation.w = 0.0

        self.hri_state = -1
        self.normal_vector = np.array([0, 0, 1])
        self.last_normal_vector = np.array([0, 0, 1])

        # Inicializar vbles
        self.kp_msg = KP_URDF()
        self.current_pose = PoseStamped()
        
        self.commanded_pose = PoseStamped()
        # self.commanded_pose.header.frame_id = "fr3_link0"
        # self.commanded_pose.pose.position.x = 0.25
        # self.commanded_pose.pose.position.y = 0.0
        # self.commanded_pose.pose.position.z = 0.5
        # self.commanded_pose.pose.orientation.x = 0.92
        # self.commanded_pose.pose.orientation.y = -0.37
        # self.commanded_pose.pose.orientation.z = 0.0
        # self.commanded_pose.pose.orientation.w = 0.0

    def obtain_hri_state_callback(self, msg):
        """
        Función callback que pone a disposición del nodo el mensaje 
        """
        rospy.loginfo("hri_state actualizado")
        self.hri_state = msg.data


    def obtain_current_pose_callback(self, msg):
        """
        Función encargada de obtener el FrankaState.O_T_EE y calcular el current_pose como PoseStamped
        """
        # Convertir FrankaState a matriz de transformación 4x4
        current_0_T_EE = np.array(msg.O_T_EE).reshape(4, 4)
        current_0_T_EE = current_0_T_EE.T # Trasponer para que sea una matriz de transformación típica.

        # Extraer posición y orientación
        current_position = current_0_T_EE[:3, 3]  # Extraer traslación (X, Y, Z)

        # DEBUG
        # rospy.loginfo(f"X: {current_0_T_EE[0, 3]}")
        # rospy.loginfo(f"Y: {current_0_T_EE[1, 3]}")
        # rospy.loginfo(f"Z: {current_0_T_EE[2, 3]}")

        current_orientation = R.from_matrix(current_0_T_EE[:3, :3])  # Extraer rotación como quaternion

        # Asignar a equilibrium_pose
        current_pose = PoseStamped()
        current_pose.header.stamp = rospy.Time.now()
        current_pose.header.frame_id = "fr3_link0"  # Cambiar si es necesario

        current_pose.pose.position.x = current_position[0]
        current_pose.pose.position.y = current_position[1]
        current_pose.pose.position.z = current_position[2]

        current_pose.pose.orientation.x = current_orientation.as_quat()[0]
        current_pose.pose.orientation.y = current_orientation.as_quat()[1]
        current_pose.pose.orientation.z = current_orientation.as_quat()[2]
        current_pose.pose.orientation.w = current_orientation.as_quat()[3]

        # Guardar la pose actual
        self.current_pose = current_pose

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

    def kp_callback(self, msg):
        """
        Función que pone a disposición del nodo los KP
        """
        self.kp_msg = msg
    

    def desired_pose_publisher_callback(self):
        """
        Callback para actualizar el equilibrium pose del Franka en función de los datos de KP_URDF.

        """
        try:
            self.desired_pose.header.stamp = rospy.Time.now()
            print_current_pose(self.current_pose)

            ## Gestión de casos
            if self.hri_state == 1:
                # Caso 1. Posición 1
                rospy.loginfo("Punto 1")
                self.desired_pose.pose.position.x = 0.7
                self.desired_pose.pose.position.y = 0.0
                self.desired_pose.pose.position.z = 0.5
                self.desired_pose.pose.orientation.x = 0.92
                self.desired_pose.pose.orientation.y = -0.37
                self.desired_pose.pose.orientation.z = 0.0
                self.desired_pose.pose.orientation.w = 0.0

                commanded_pose = smooth_interpolation(self.current_pose, self.desired_pose, max_step=0.1)
                self.pub.publish(commanded_pose)

            elif self.hri_state == 2:
                rospy.loginfo("Aproximando...")
                # Aproximación a la muñeca
                # Consignas. 
                # - La longitud del antebrazo no debe cambiar!!!!!!!!!!!!!!!!!!!!!
                # - La herramienta F y el EE están girados 45 sobre Z.

                magnitud = 0.1
                vect_forearm = np.array([self.kp_msg.right_wrist.x, self.kp_msg.right_wrist.y, self.kp_msg.right_wrist.z]) - \
                               np.array([self.kp_msg.right_elbow.x, self.kp_msg.right_elbow.y, self.kp_msg.right_elbow.z])

                # Calcular posición de F.
                # normal_vector debe estar normalizado para aplicar la posición correctamente.
                self.desired_pose.pose.position.x = self.kp_msg.right_wrist.x + self.normal_vector[0]*magnitud
                self.desired_pose.pose.position.y = self.kp_msg.right_wrist.y + self.normal_vector[1]*magnitud
                self.desired_pose.pose.position.z = self.kp_msg.right_wrist.z + self.normal_vector[2]*magnitud

                # Calcular orientación de F respecto a 0. F es nuestra herramienta personalizada. 0 es la base del robot.
                # Los ejes de F se alinean como:
                #   - Eje Y de F con vect_forearm
                #   - Eje Z de F con -normal_vector pq queremos que se aproxime hacia el brazo.
                self.desired_pose.pose.orientation = calculate_quaternion_0_F(vect_forearm, -self.normal_vector)

                commanded_pose = smooth_interpolation(self.current_pose, self.desired_pose, max_step=1)
                self.pub.publish(commanded_pose)
                
            elif self.hri_state == 3:
                # Agarrando
                rospy.loginfo("Agarrando..")

            elif self.hri_state == 4:
                # Soltando
                # Monitorizar la posición del efector final, si se sale de cierto espacio, vuelve al reposo.
                rospy.loginfo("Soltando...")

            else:
                # Caso 0. Posición 0
                rospy.loginfo("En reposo")
                self.desired_pose.pose.position.x = 0.25
                self.desired_pose.pose.position.y = 0.0
                self.desired_pose.pose.position.z = 0.5
                self.desired_pose.pose.orientation.x = 0.92
                self.desired_pose.pose.orientation.y = -0.37
                self.desired_pose.pose.orientation.z = 0.0
                self.desired_pose.pose.orientation.w = 0.0

                commanded_pose = smooth_interpolation(self.current_pose, self.desired_pose, max_step=0.1)
                self.pub.publish(commanded_pose)
        
            rospy.logdebug("Actualizando equilibrium_pose")

        except Exception as e:
            rospy.logwarn(f"Error en eq_pose_publisher_callback: {e}")

if __name__ == '__main__':
    try:
        eq_publisher = EquilibriumPosePublisher()

        # Bucle de control
        rate = rospy.Rate(50)  # 10 Hz
        while not rospy.is_shutdown():
            eq_publisher.desired_pose_publisher_callback()
            rate.sleep()
        
        # Salida controlada del nodo
        def shutdown_callback():
            rospy.loginfo("Shutting down equilibrium_pose_publisher node...")
        rospy.on_shutdown(shutdown_callback)

    except rospy.ROSInterruptException:
        pass