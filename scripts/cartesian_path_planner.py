#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
import copy
from scipy.spatial.transform import Rotation as R, Slerp

class CartesianPathPlanner:
    def __init__(self):
        rospy.init_node("cartesian_path_planner", log_level=rospy.DEBUG)

        # Publicadores
        self.equilibrium_pose_publisher = rospy.Publisher("/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=10)
        self.path_planner_state_publisher = rospy.Publisher("/path_planner_state", Int32, queue_size=2)

        # Subscriptores
        rospy.Subscriber('/desired_pose', PoseStamped, self.obtain_desired_pose_callback, queue_size=1) # Subscripción a la pose deseada
        rospy.Subscriber('/current_pose', PoseStamped, self.obtain_current_pose_callback, queue_size=10) # Subscripción a la pose actual
        
        # Inicialización de variables
        self.current_pose = PoseStamped()
        self.current_pose.header.stamp = rospy.Time(0)  # Indicar que no tiene datos aún
        
        self.desired_pose = PoseStamped()
        self.desired_pose.header.stamp = rospy.Time(0)
        
        self.last_desired_pose = PoseStamped()  # Usar PoseStamped vacío
        self.last_desired_pose.header.stamp = rospy.Time(0)  # Indicar que aún no tiene valor

        self.path_planner_state_publisher.publish(0) # Estado inicial
        
        rospy.sleep(1) # Pausa para inicializar


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

    
    def send_equilibrium_pose(self, initial_pose, target_pose, duration=1.0, rate_hz=100, threshold=0.005):
        """ 
        Envía la pose deseada con interpolación lineal en posición y SLERP en orientación.
        Se detiene si la diferencia entre la pose actual y la deseada está por debajo del threshold.
        """
        
        rospy.logdebug("Path_planner: Iniciando planificación de trayectoria")
        
        # Validación de poses
        if initial_pose.header.stamp == rospy.Time(0):
            rospy.logwarn("Path_planner: Pose inicial inválida. Abortando planificación.")
            return

        # Configurar interpolación
        steps = int(duration * rate_hz)
        rate_hz = min(max(steps / duration, 10), 200)  # Limita entre 10 y 200 Hz
        rate = rospy.Rate(rate_hz)

        # Obtener posiciones inicial y final
        init_pos = np.array([initial_pose.pose.position.x, initial_pose.pose.position.y, initial_pose.pose.position.z])
        goal_pos = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z])
        
        # Interpolación de posiciones
        trajectory_pos = np.linspace(init_pos, goal_pos, steps)

        # Obtener cuaterniones de orientación inicial y final
        init_quat = R.from_quat([initial_pose.pose.orientation.x, 
                                initial_pose.pose.orientation.y, 
                                initial_pose.pose.orientation.z, 
                                initial_pose.pose.orientation.w])
        
        goal_quat = R.from_quat([target_pose.pose.orientation.x, 
                                target_pose.pose.orientation.y, 
                                target_pose.pose.orientation.z, 
                                target_pose.pose.orientation.w])
        
        rospy.loginfo(f"Generados {len(trajectory_pos)} puntos en la trayectoria.")

        # Interpolación de orientaciones

        # Definir los tiempos de interpolación
        times = [0, 1] # Esto es arbitrario. Se usa para normalizar con steps. Al final, el tiempo de 0.5 es equivalente al 0.5*steps

        # Se crea el objeto Slerp. Esto no aplica la inteporlación esferia, sino q asocia tiempo y cuaternion. Posteriormente se interpola en el tiempo.
        slerp = Slerp(times, R.from_quat([init_quat.as_quat(), goal_quat.as_quat()]))
        trajectory_quat = slerp(np.linspace(0, 1, steps)) # Se espacia para el número de pasos
        
        if len(trajectory_pos) == 0:
            rospy.logerr("ERROR: No se generaron puntos en la trayectoria. Verifica los valores de start y end.")
            return

        # Publicar poses
        for i in range(steps):
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "fr3_link0"
            
            # Asignar posición interpolada
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = trajectory_pos[i]

            # Asignar orientación interpolada con SLERP
            quat = trajectory_quat[i].as_quat()
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat
            
            rospy.logdebug(f"Publicando equilibrium_pose: x={pose_msg.pose.position.x:.4f}, y={pose_msg.pose.position.y:.4f}, z={pose_msg.pose.position.z:.4f}")

            self.path_planner_state = 1
            self.equilibrium_pose_publisher.publish(pose_msg)
            self.path_planner_state_publisher.publish(self.path_planner_state)

            # Verificación de finalización antes de completar la trayectoria
            current_pos = np.array([self.current_pose.pose.position.x, 
                                    self.current_pose.pose.position.y, 
                                    self.current_pose.pose.position.z])
            
            # Distancia entre posición actual y deseada
            distance_error = np.linalg.norm(goal_pos - current_pos)

            if distance_error < threshold:
                rospy.loginfo(f"Trayectoria completada antes de tiempo. Error final: {distance_error:.6f}")
                break  # Terminar antes si se ha alcanzado la pose deseada

            rate.sleep()
        
        # Movimiento completado
        self.path_planner_state = 2
        self.path_planner_state_publisher.publish(self.path_planner_state)


if __name__ == "__main__":

    rospy.loginfo("Nodo iniciado correctamente")
    planner = CartesianPathPlanner()

    # while (planner.desired_pose == None):
    #     rospy.sleep(1)

    rospy.loginfo("Esperando la primera pose deseada...")
    planner.desired_pose = rospy.wait_for_message('/desired_pose', PoseStamped)


    while not rospy.is_shutdown(): # Mantiene un bucle constante

        # Verificar si hay una nueva pose deseada
        if planner.desired_pose and (planner.last_desired_pose.header.stamp == rospy.Time(0) or
                             planner.desired_pose.pose.position != planner.last_desired_pose.pose.position):


            # # Estado 0. Esperando nueva pose deseada
            planner.path_planner_state = 0
            planner.path_planner_state_publisher.publish(0)

            rospy.loginfo("Nueva pose detectada, iniciando movimiento...")
            
            # Esperar pose actual válida
            start_time = rospy.Time.now()
            while planner.current_pose.header.stamp == rospy.Time(0):
                rospy.logwarn("Esperando pose actual...")
                if (rospy.Time.now() - start_time).to_sec() > 5:  # Timeout de 5 segundos
                    rospy.logerr("Timeout esperando pose actual. Abortando movimiento.")
                    break
                rospy.sleep(1)

            initial_pose = copy.deepcopy(planner.current_pose)
            planner.last_desired_pose = copy.deepcopy(planner.desired_pose)

            planner.send_equilibrium_pose(initial_pose, planner.desired_pose)
        
        rospy.sleep(1)  # Controlar el ciclo del loop

