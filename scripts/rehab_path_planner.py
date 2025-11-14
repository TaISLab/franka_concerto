#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
import copy
from scipy.spatial.transform import Rotation as R, Slerp
import scipy.interpolate

import scipy.interpolate
import time

# Publicar Eq pose a mano. Pon en terminal esto:
# rostopic pub /desired_pose geometry_msgs/PoseStamped "header:
#   stamp: now
#   frame_id: 'fr3_link0'
# pose:
#   position:
#     x: 0.5
#     y: 0.0
#     z: 0.3
#   orientation:
#     x: 1.0
#     y: 0.0
#     z: 0.0
#     w: 0.0"


class CartesianPathPlanner:
    def __init__(self):
        rospy.init_node("rehab_path_planner_node", log_level=rospy.DEBUG)

        # Publicadores
        self.equilibrium_pose_publisher = rospy.Publisher("/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=10)
        self.path_planner_state_publisher = rospy.Publisher("/rehab_path/state", Int32, queue_size=2)

        # Subscriptores
        rospy.Subscriber('/rehab_path/desired_pose', PoseStamped, self.obtain_desired_pose_callback, queue_size=1) # Subscripción a la pose deseada
        rospy.Subscriber('/current_pose', PoseStamped, self.obtain_current_pose_callback, queue_size=10) # Subscripción a la pose actual
        rospy.Subscriber('/fusion_model')

        # Inicialización de variables
        self.current_pose = PoseStamped()
        self.current_pose.header.stamp = rospy.Time(0)  # Indicar que no tiene datos aún
        
        self.desired_pose = PoseStamped()
        self.desired_pose.header.stamp = rospy.Time(0)
        
        self.last_desired_pose = PoseStamped()  # Usar PoseStamped vacío
        self.last_desired_pose.header.stamp = rospy.Time(0)  # Indicar que aún no tiene valor

        self.path_planner_state_publisher.publish(0) # ESTADO_INICIAL
        
        rospy.sleep(1) # Pausa para inicializar

    # Limitación del espacio de trabajo
    # def esta_dentro_espacio_trabajo(self, pose):
    #     """
    #     Verifica si una pose está dentro del espacio de trabajo definido como un volumen. El origen es fr3_link0, la base del manipulador.
    #     Los límites son:
    #     :param pose: PoseStamped con la posición del punto.

    #     :return: True si está dentro, False si está fuera.
    #     """
    #     x_limits = (-0.35, 0.90)
    #     y_limits = (-0.50, 0.90)
    #     z_limits = (0.10, 0.90)

    #     x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
    #     return (x_limits[0] <= x <= x_limits[1] and
    #             y_limits[0] <= y <= y_limits[1] and
    #             z_limits[0] <= z <= z_limits[1])


    def print_pose(self, pose_stamped, description="Pose"):
        """
        Imprime la información de la pose en la terminal.
        """
        rospy.loginfo(f"{description}: x={pose_stamped.pose.position.x:.4f}, "
                      f"y={pose_stamped.pose.position.y:.4f}, "
                      f"z={pose_stamped.pose.position.z:.4f}")

    def interpolate_trajectory(self, start, end, steps=50):
        """Genera una trayectoria interpolada entre start y end en 'steps' pasos."""
        traj = np.linspace(start, end, steps)
        return traj

    def obtain_current_pose_callback(self, msg):
        """ Callback que obtiene la pose actual del robot desde el nuevo controlador """
        self.current_pose = msg

    def obtain_desired_pose_callback(self, msg):
        """ Callback que obtiene la pose deseada para el robot"""
        # self.desired_pose = PoseStamped()
        # self.desired_pose.header.stamp = rospy.Time.now()
        # self.desired_pose.header.frame_id = "fr3_link0"
        # self.desired_pose.pose = msg
        
        # self.desired_pose = msg
        self.desired_pose = copy.deepcopy(msg)  # Copia segura del mensaje
    
    def send_equilibrium_pose(self, initial_pose, target_pose, duration=5.0, rate_hz=30, threshold=0.03, max_iterations=1):
        """ 
        Control en lazo cerrado con realimentación de la pose actual.
        Se recalcula la interpolación si hay un error significativo.
        Ahora con interpolación de Splines Cúbicos.
        """
        
        rospy.logdebug("Path_planner: Iniciando planificación de trayectoria con Splines Cúbicos")
        
        if initial_pose.header.stamp == rospy.Time(0):
            rospy.logwarn("Path_planner: Pose inicial inválida. Abortando planificación.")
            self.path_planner_state_publisher.publish(-2) # ERROR_GENERICO
            return
        
        rate = rospy.Rate(rate_hz)
        iteration = 0
        
        while iteration < max_iterations:
            initial_pose = copy.deepcopy(self.current_pose)
            init_pos = np.array([initial_pose.pose.position.x, initial_pose.pose.position.y, initial_pose.pose.position.z])
            goal_pos = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z])

            # Interpolación con Splines Cúbicos
            time_steps = np.linspace(0, 1, int(duration * rate_hz))
            cubic_spline = scipy.interpolate.CubicSpline([0, 1], np.vstack([init_pos, goal_pos]), axis=0)
            trajectory_pos = cubic_spline(time_steps)
            
            # Obtener cuaterniones inicial y final
            init_quat = R.from_quat([
                initial_pose.pose.orientation.x, 
                initial_pose.pose.orientation.y, 
                initial_pose.pose.orientation.z, 
                initial_pose.pose.orientation.w])
            goal_quat = R.from_quat([
                target_pose.pose.orientation.x, 
                target_pose.pose.orientation.y, 
                target_pose.pose.orientation.z, 
                target_pose.pose.orientation.w])
            
            # SLERP para interpolación de cuaterniones con más puntos
            slerp = Slerp([0, 1], R.from_quat([init_quat.as_quat(), goal_quat.as_quat()]))
            trajectory_quat = slerp(time_steps)
            
            for i in range(len(trajectory_pos)):
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "fr3_link0"
                
                # Asignar posición interpolada
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = trajectory_pos[i]
                
                # Asignar orientación interpolada con SLERP
                quat = trajectory_quat[i].as_quat()
                pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat
                
                self.equilibrium_pose_publisher.publish(pose_msg)
                self.path_planner_state_publisher.publish(1) # ESTADO_EJECUTANDOSE
                
                current_pos = np.array([self.current_pose.pose.position.x, 
                                        self.current_pose.pose.position.y, 
                                        self.current_pose.pose.position.z])
                distance_error = np.linalg.norm(goal_pos - current_pos)
                
                if distance_error < threshold:
                    rospy.loginfo(f"Trayectoria completada con error final: {distance_error:.6f}")
                    self.path_planner_state_publisher.publish(2) # ESTADO_COMPLETADO
                    return
                
                rate.sleep()
            
            rospy.logwarn(f"Iteración {iteration+1}: Error aún presente ({distance_error:.6f}). Recalculando trayectoria...")
            iteration += 1
        
        rospy.logerr("No se pudo alcanzar la pose deseada tras múltiples iteraciones.")
        self.path_planner_state_publisher.publish(3) # ESTADO_COMPLETADO_ERROR_CARTESIANO

if __name__ == "__main__":

    rospy.loginfo("Nodo iniciado correctamente")
    planner = CartesianPathPlanner()

    planner.path_planner_state_publisher.publish(0) # ESTADO_INICIAL

    rospy.logdebug("Esperando la primera pose deseada...")
    planner.desired_pose = rospy.wait_for_message('/cartesian_path/desired_pose', PoseStamped)

    while not rospy.is_shutdown(): # Mantiene un bucle constante
        
        # DEBUG
        # rospy.logdebug(f"last_desired_pose: x={planner.last_desired_pose.pose.position.x:.4f}, y={planner.last_desired_pose.pose.position.y:.4f}, z={planner.last_desired_pose.pose.position.z:.4f}")
        rospy.logdebug(f"desired_pose: x={planner.desired_pose.pose.position.x:.4f}, y={planner.desired_pose.pose.position.y:.4f}, z={planner.desired_pose.pose.position.z:.4f}")

        
        
        # Verificar si hay una nueva pose deseada diferente
        if planner.desired_pose and (not (np.isclose(planner.desired_pose.pose.position.x, planner.last_desired_pose.pose.position.x, atol=0.01) and
                                          np.isclose(planner.desired_pose.pose.position.y, planner.last_desired_pose.pose.position.y, atol=0.01) and
                                          np.isclose(planner.desired_pose.pose.position.z, planner.last_desired_pose.pose.position.z, atol=0.01) and
                                          np.isclose(planner.desired_pose.pose.orientation.x, planner.last_desired_pose.pose.orientation.x, atol=0.01) and
                                          np.isclose(planner.desired_pose.pose.orientation.y, planner.last_desired_pose.pose.orientation.y, atol=0.01) and
                                          np.isclose(planner.desired_pose.pose.orientation.z, planner.last_desired_pose.pose.orientation.z, atol=0.01) and
                                          np.isclose(planner.desired_pose.pose.orientation.w, planner.last_desired_pose.pose.orientation.w, atol=0.01))):
            
            planner.path_planner_state_publisher.publish(0) # ESTADO_INICIAL

            # Verificar que esté dentro del volumen de trabajo
            if planner.esta_dentro_espacio_trabajo(planner.desired_pose):
                
                # Esperar pose actual válida
                start_time = rospy.Time.now()
                while planner.current_pose.header.stamp == rospy.Time(0):
                    rospy.logwarn("Esperando pose actual...")
                    if (rospy.Time.now() - start_time).to_sec() > 5:  # Timeout de 5 segundos
                        rospy.logerr("Timeout esperando pose actual. Abortando movimiento.")
                        planner.path_planner_state_publisher.publish(-2) # ERROR_GENERICO
                        break
                    rospy.sleep(1)

                initial_pose = copy.deepcopy(planner.current_pose)
                planner.last_desired_pose = copy.deepcopy(planner.desired_pose)

                rospy.loginfo("Path_planner. Nueva pose recibida, iniciando movimiento...")
                planner.send_equilibrium_pose(initial_pose, planner.desired_pose)

            else:
                rospy.logerr("Pose fuera del espacio de trabajo. Abortando...")
                planner.path_planner_state_publisher.publish(-3) # ERROR_LIMITE_CARTESIANO

        rospy.sleep(1)  # Controlar el ciclo del loop

