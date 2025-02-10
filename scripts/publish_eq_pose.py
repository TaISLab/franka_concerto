#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from human_articular_space.msg import KP_URDF
from std_msgs.msg import Int32  # Importa el tipo de mensaje Int32
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP
from scipy.spatial.transform import Rotation as R


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

        # Inicializar pose con valores por defecto
        self.eq_pose = PoseStamped()
        self.eq_pose.header.frame_id = "fr3_link0"
        self.eq_pose.pose.position.x = 0.25
        self.eq_pose.pose.position.y = 0.0
        self.eq_pose.pose.position.z = 0.5
        self.eq_pose.pose.orientation.x = 0.92
        self.eq_pose.pose.orientation.y = -0.37
        self.eq_pose.pose.orientation.z = 0.0
        self.eq_pose.pose.orientation.w = 0.0

        self.hri_state = -1
        self.normal_vector = np.array([0, 0, 1])

        # iniciar mensaje kp
        self.kp_msg = KP_URDF()


    def obtain_hri_state_callback(self, msg):
        """
        Función callback que pone a disposición del nodo el mensaje 
        """
        rospy.loginfo("hri_state actualizado")
        self.hri_state = msg.data

    def obtain_skeleton3D_callback(self, msg):
        """
        Función callback que calcula el vector normal al RShoulder-RElbow-RWrist
        """
        keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])

        kp_R_Shoulder = keypointsX[6]
        kp_R_Elbow = keypointsX[8]
        kp_R_Wrist = keypointsX[10]

        if are_kp_valid(kp_R_Shoulder, kp_R_Elbow, kp_R_Wrist):
            self.normal_vector = calculate_normal_vector(kp_R_Shoulder, kp_R_Elbow, kp_R_Wrist)

            # Verificar si la normal contiene valores NaN
            if np.isnan(self.normal_vector).any():
                rospy.logwarn("Se detectó NaN en el cálculo de la normal. Se mantiene el vector por defecto.")
                self.normal_vector = np.array([0, 0, 1])  # Vector por defecto

        else:
            rospy.logwarn("Keypoints inválidos. Se mantiene el vector por defecto.")
            self.normal_vector = np.array([0, 0, 1])  # Vector por defecto


    def kp_callback(self, msg):
        self.kp_msg = msg
    
    
    def eq_pose_publisher_callback(self):
        """
        Callback para actualizar el equilibrium pose del Franka en función de los datos de KP_URDF.

        Entrada:
        - Punto de la muñeca

        Salida:
        - Equilibrium_Pose

        """
        try:
            self.eq_pose.header.stamp = rospy.Time.now()

            if self.hri_state == 1:
                rospy.loginfo("En reposo")
                self.eq_pose.pose.position.x = 0.7
                self.eq_pose.pose.position.y = 0.0
                self.eq_pose.pose.position.z = 0.5
                self.eq_pose.pose.orientation.x = 0.92
                self.eq_pose.pose.orientation.y = -0.37
                self.eq_pose.pose.orientation.z = 0.0
                self.eq_pose.pose.orientation.w = 0.0

                self.pub.publish(self.eq_pose)

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
                self.eq_pose.pose.position.x = self.kp_msg.right_wrist.x + self.normal_vector[0]*magnitud
                self.eq_pose.pose.position.y = self.kp_msg.right_wrist.y + self.normal_vector[1]*magnitud
                self.eq_pose.pose.position.z = self.kp_msg.right_wrist.z + self.normal_vector[2]*magnitud

                # Calcular orientación de F respecto a 0. F es nuestra herramienta personalizada. 0 es la base del robot.
                # Los ejes de F se alinean como:
                #   - Eje Y de F con vect_forearm
                #   - Eje Z de F con -normal_vector pq queremos que se aproxime hacia el brazo.

                self.eq_pose.pose.orientation = calculate_quaternion_0_F(vect_forearm, -self.normal_vector)
                self.pub.publish(self.eq_pose)
                
            elif self.hri_state == 6:
                # Agarrando
                rospy.loginfo("Agarrando")

                self.eq_pose.pose.position.x = self.kp_msg.right_wrist.x
                self.eq_pose.pose.position.y = self.kp_msg.right_wrist.y
                self.eq_pose.pose.position.z = self.kp_msg.right_wrist.z + 0.1

                self.pub.publish(self.eq_pose)

            
            elif self.hri_state == 3:
                rospy.loginfo("Soltando...")
                # Soltando
                # Monitorizar la posición del efector final, si se sale de cierto espacio, vuelve al reposo.

            else:
                # Reposo
                rospy.loginfo("En reposo")
                self.eq_pose.pose.position.x = 0.25
                self.eq_pose.pose.position.y = 0.0
                self.eq_pose.pose.position.z = 0.5
                self.eq_pose.pose.orientation.x = 0.92
                self.eq_pose.pose.orientation.y = -0.37
                self.eq_pose.pose.orientation.z = 0.0
                self.eq_pose.pose.orientation.w = 0.0

                self.pub.publish(self.eq_pose)
        
            rospy.loginfo("Actualizando equilibrium_pose")

        except Exception as e:
            rospy.logwarn(f"Error en eq_pose_publisher_callback: {e}")

if __name__ == '__main__':
    try:
        eq_publisher = EquilibriumPosePublisher()

        def shutdown_callback():
            rospy.loginfo("Shutting down equilibrium_pose_publisher node...")
        rospy.on_shutdown(shutdown_callback)

        # Bucle de control
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            eq_publisher.eq_pose_publisher_callback()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass