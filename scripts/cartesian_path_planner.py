#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Quaternion

class CartesianPathPlanner:
    def __init__(self):
        rospy.init_node("cartesian_path_planner")

        # Publicador
        self.equilibrium_pose_publisher = rospy.Publisher("/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=10)
        # self.current_pose_publisher = rospy.Publisher("/current_pose", PoseStamped, queue_size=10)
        # self.equilibrium_pose_publisher = rospy.Publisher("/path_planner/pos_desired", PoseStamped, queue_size=10)

        # Subscribers
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.obtain_current_pose_callback)  # Subscripción al estado del robot
        rospy.Subscriber('/pose_desired', PoseStamped, self.obtain_desired_pose_callback) # Subscripción a la pose deseada

        # Inicializar pose actual como vacía
        self.current_pose = PoseStamped()
        self.current_pose.header.stamp = rospy.Time(0)  # Indicar que no tiene datos aún

        self.initial_pose = PoseStamped()
        self.desired_pose = PoseStamped()
        self.desired_pose.header.stamp = rospy.Time(0)  # No tiene datos aún
        
        rospy.sleep(1)


    def print_current_pose(self, pose_stamped, description="Current Pose"):
        """
        Imprime la información de la pose actual en la terminal.
        """
        rospy.loginfo(f"{description}: x={pose_stamped.pose.position.x:.4f}, "
                      f"y={pose_stamped.pose.position.y:.4f}, "
                      f"z={pose_stamped.pose.position.z:.4f}")

    def interpolate_trajectory(self, start, end, steps=50):
        """Genera una trayectoria interpolada entre start y end en 'steps' pasos."""
        traj = np.linspace(start, end, steps)
        return traj

    def obtain_current_pose_callback(self, msg):
        """ Callback que obtiene la pose actual del robot desde FrankaState """
        current_0_T_EE = np.array(msg.O_T_EE).reshape(4, 4).T  # Trasponer la matriz

        # Extraer posición y orientación
        current_position = current_0_T_EE[:3, 3]
        current_orientation = R.from_matrix(current_0_T_EE[:3, :3])  # Quaternion

        # Asignar la pose
        self.current_pose.header.stamp = rospy.Time.now()
        self.current_pose.header.frame_id = "fr3_link0"
        self.current_pose.pose.position.x, self.current_pose.pose.position.y, self.current_pose.pose.position.z = current_position
        self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y, self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w = current_orientation.as_quat()

    def obtain_desired_pose_callback(self, msg):
        """ Callback que obtiene la pose deseada para el robot"""

        self.desired_pose.header.stamp = rospy.Time.now()
        self.desired_pose.header.frame_id = "fr3_link0"
        self.desired_pose.pose.position = msg.pose.position
        self.desired_pose.pose.orientation = msg.pose.orientation

    
    def send_equilibrium_pose(self, initial_pose, target_pose, duration=10.0, rate_hz=100):
        """ Envía la pose deseada con interpolación suave """
        # rospy.loginfo("Iniciando el envío de equilibrium_pose...")
        
        # Esperar a recibir una pose válida
        while initial_pose.header.stamp == rospy.Time(0):  
            rospy.logwarn("Esperando a recibir la pose actual del robot...")
            rospy.sleep(0.1)

        rospy.loginfo("Recibida pose inicial, comenzando interpolación")
        
        # Configurar rate correctamente
        rate = rospy.Rate(rate_hz)  # `rate_hz` es un número, mientras que `rate` es un objeto Rate
        steps = int(duration * rate_hz)  # ← Ahora funciona bien


        # Obtener posiciones inicial y final
        init_pose = np.array([initial_pose.pose.position.x, 
                              initial_pose.pose.position.y, 
                              initial_pose.pose.position.z])
        goal_pose = np.array([target_pose.pose.position.x, 
                              target_pose.pose.position.y, 
                              target_pose.pose.position.z])
        

        # rospy.loginfo("Inicio de interpolación")

        # Generar trayectorias interpoladas
        rospy.logwarn(f"init_pose: x={init_pose[0]:.4f}, y={init_pose[1]:.4f}, z={init_pose[2]:.4f}")
        rospy.logwarn(f"goal_pose: x={goal_pose[0]:.4f}, y={goal_pose[1]:.4f}, z={goal_pose[2]:.4f}")

        trajectory = self.interpolate_trajectory(init_pose, goal_pose, steps)
        # rospy.loginfo("Interpolación finalizada")

        rospy.loginfo(f"Generados {len(trajectory)} puntos en la trayectoria.")
        if len(trajectory) == 0:
            rospy.logerr("ERROR: No se generaron puntos en la trayectoria. Verifica los valores de start y end.")

        # Publicar poses
        for point in trajectory:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "fr3_link0"
            pose_msg.pose.position.x = point[0]
            pose_msg.pose.position.y = point[1]
            pose_msg.pose.position.z = point[2]
            pose_msg.pose.orientation = target_pose.pose.orientation

            rospy.logwarn(f"Publicando equilibrium_pose: x={pose_msg.pose.position.x:.4f}, y={pose_msg.pose.position.y:.4f}, z={pose_msg.pose.position.z:.4f}")
            
            self.equilibrium_pose_publisher.publish(pose_msg)  # Intentando publicar
        
            rate.sleep()


if __name__ == "__main__":
    rospy.loginfo("Nodo iniciado correctamente")
    planner = CartesianPathPlanner()

    # Esperar a obtener la pose actual
    while planner.current_pose.header.stamp == rospy.Time(0):  
        rospy.logwarn("Esperando a recibir la pose actual del robot...")
    rospy.sleep(0.1)
    planner.initial_pose = planner.current_pose

    # Esperar a obtener la pose deseada
    while planner.desired_pose.header.stamp == rospy.Time(0):  
        rospy.logwarn("Esperando a recibir la pose deseada del robot...")

    # Asignación de pose
    target_pose = PoseStamped()
    target_pose.pose = planner.desired_pose.pose 

    rospy.sleep(2)  # Esperar que todo esté listo antes de enviar
    planner.send_equilibrium_pose(planner.initial_pose, target_pose)

    rospy.spin()
