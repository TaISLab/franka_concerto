#!/usr/bin/env python3

import rospy
import numpy as np
import copy
import time
from scipy.spatial.transform import Rotation as R, Slerp
import scipy.interpolate

# --- Imports de ROS Actionlib ---
import actionlib
from franka_concerto.msg import MoveFR3Action, MoveFR3Goal, MoveFR3Result, MoveFR3Feedback
from actionlib_msgs.msg import GoalStatusArray
from franka_msgs.msg import FrankaState

# --- Imports de Mensajes ROS ---
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from std_msgs.msg import Int32

# --- Imports para cambiar controladores ---
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, SwitchControllerResponse
from controller_manager_msgs.srv import LoadController, LoadControllerRequest
from control_msgs.msg import FollowJointTrajectoryResult, FollowJointTrajectoryActionResult

class Fr3ActionServer:
    
    # --- CONSTANTES ---
    # Nombres de controladores (ajusta según tu config .yaml)
    VELOCITY_CONTROLLER_NAME = "cartesian_velocity_external_controller"
    IMPEDANCE_CONTROLLER_NAME = "cartesian_impedance_example_controller" 
    EFFORT_CONTROLLER_NAME = "effort_joint_trajectory_controller"
    
    # Parámetros de Homing (¡Ajústalos a tu robot!)
    HOME_POSE_POSITION = np.array([0.5, 0.0, 0.3]) # x, y, z
    HOME_POSE_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0]) # x, y, z, w (apuntando hacia abajo)
    HOMING_MAX_SPEED = 0.04  # m/s (¡Límite de velocidad para homing!)
    HOMING_P_GAIN = 0.8     # Ganancia proporcional para el control de homing
    HOMING_POS_THRESHOLD = 0.01 # 1 cm
    
    def __init__(self):
        rospy.init_node("fr3_motion_action_server")

        # --- Variables de estado ---
        self.current_pose = PoseStamped()
        self.current_pose.header.stamp = rospy.Time(0)  # Indicar que no tiene datos aún

        # --- Publicadores ---
        # Publicador para el controlador de IMPEDANCIA
        self.impedance_pose_pub = rospy.Publisher(
            f"/cartesian_impedance_example_controller/equilibrium_pose", 
            PoseStamped, 
            queue_size=10
        )
        # Publicador para el controlador de VELOCIDAD
        self.velocity_cmd_pub = rospy.Publisher(
            f"/robot_vel_ctrl/vel_cmd", 
            TwistStamped, 
            queue_size=10
        )
        self.v_commanded = np.zeros(6)
        # --- Subscriptores ---
        rospy.Subscriber('/franka_state_controller/ee_pose', PoseStamped, self.obtain_current_pose_callback, queue_size=10)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_state_callback, queue_size=10)
        # --- Cliente de Servicio para Cambiar Controladores ---
        rospy.loginfo("Esperando al servicio del Controller Manager...")
        try:
            rospy.wait_for_service('/controller_manager/switch_controller', timeout=5.0)
            self.switch_controller_srv = rospy.ServiceProxy(
                '/controller_manager/switch_controller', 
                SwitchController
            )
            rospy.loginfo("¡Conectado al Controller Manager!")
        except (rospy.ServiceException, rospy.ROSException, rospy.ROSInterruptException) as e:
            rospy.logerr(f"No se pudo conectar al Controller Manager: {e}")
            rospy.signal_shutdown("Error crítico del servicio")
            return

        # --- Inicializar el Servidor de Acciones ---
        self._feedback = MoveFR3Feedback()
        self._result = MoveFR3Result()

        self._as = actionlib.SimpleActionServer(
            "fr3_motion_server",  # Nombre de la acción (el cliente lo usará)
            MoveFR3Action,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self._as.start()
        rospy.loginfo("Servidor de Acciones 'fr3_motion_server' iniciado.")

        self.effort_result = None
        self.effort_status = None
        rospy.Subscriber(
            "/effort_joint_trajectory_controller/follow_joint_trajectory/result",
            FollowJointTrajectoryActionResult,
            self.effort_result_callback,
            queue_size=1
        )
        rospy.Subscriber(
            "/effort_joint_trajectory_controller/follow_joint_trajectory/status",
            GoalStatusArray,
            self.effort_status_callback,
            queue_size=1
        )

    def franka_state_callback(self, msg):
        """Actualiza self.current_velocity con FrankaState.O_dP_EE_c (EE twist comandado en base)."""
        try:
            v_commanded = msg.O_dP_EE_c

            if v_commanded is None:
                # Si la definición del mensaje difiere, lo reportamos para debugging
                rospy.logwarn("FrankaState no contiene O_dP_EE_c")
                return
            # Asegurarse de convertir a numpy array de longitud 6
            self.v_commanded = np.array(v_commanded, dtype=float)
            

        except Exception as e:
            rospy.logwarn(f"Error leyendo FrankaState.O_dP_EE_c: {e}")

    def obtain_current_pose_callback(self, msg):
        """ Callback que obtiene la pose actual del robot """
        self.current_pose = msg

    def effort_result_callback(self, msg):
        self.effort_result = msg

    def effort_status_callback(self, msg):
        # Guarda el último status recibido (puedes filtrar por goal_id si lo necesitas)
        if msg.status_list:
            self.effort_status = msg.status_list[-1].status  # Último status recibido
    
    def _ramp_down_velocity(self, rate_hz=100.0, alpha=0.9, timeout=3.0, tol=1e-3):
        """
        Hace una deceleración suave desde la velocidad comandada actual (self.v_commanded)
        hasta ~0, publicando TwistStamped en /robot_vel_ctrl/vel_cmd.

        - alpha en (0,1): más pequeño = frena más rápido.
        - timeout: tiempo máximo en segundos.
        """
        rate = rospy.Rate(rate_hz)
        start_t = rospy.Time.now()

        while (rospy.Time.now() - start_t).to_sec() < timeout and not rospy.is_shutdown():
            current_vel = np.array(self.v_commanded, dtype=float)

            # Si ya estamos prácticamente parados, salir
            if np.allclose(current_vel, np.zeros(6), atol=tol):
                break

            # Paso de deceleración exponencial
            current_vel *= alpha

            ts = TwistStamped()
            ts.header.stamp = rospy.Time.now()
            ts.twist.linear.x  = float(current_vel[0])
            ts.twist.linear.y  = float(current_vel[1])
            ts.twist.linear.z  = float(current_vel[2])
            ts.twist.angular.x = float(current_vel[3])
            ts.twist.angular.y = float(current_vel[4])
            ts.twist.angular.z = float(current_vel[5])

            self.velocity_cmd_pub.publish(ts)
            rate.sleep()


    def _ensure_controller_loaded(self, controller_name):
        """
        Carga el controlador si no está presente en controller_manager.
        """
        try:
            rospy.wait_for_service('/controller_manager/load_controller', timeout=5.0)
            load_srv = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
            resp = load_srv(controller_name)
            if resp.ok:
                rospy.loginfo(f"Controlador '{controller_name}' cargado correctamente.")
                return True
            else:
                rospy.logerr(f"No se pudo cargar el controlador '{controller_name}'.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Error al cargar el controlador '{controller_name}': {e}")
            return False

    def _get_controller_states(self):
        """
        Devuelve un dict {nombre: estado} de todos los controladores cargados.
        """
        try:
            from controller_manager_msgs.srv import ListControllers
            rospy.wait_for_service('/controller_manager/list_controllers', timeout=5.0)
            list_srv = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
            resp = list_srv()
            return {c.name: c.state for c in resp.controller}
        except Exception as e:
            rospy.logerr(f"No se pudo obtener la lista de controladores: {e}")
            return {}

    def _switch_controller(self, start_controller_name):
        """
        Cambia al controlador deseado, cargándolo y arrancándolo si es necesario.
        Solo detiene controladores que estén 'running'.
        """
        controllers = self._get_controller_states()
        # Cargar si no existe
        if start_controller_name not in controllers:
            rospy.logwarn(f"El controlador '{start_controller_name}' no está cargado. Intentando cargarlo...")
            if not self._ensure_controller_loaded(start_controller_name):
                return False
            controllers = self._get_controller_states()

        # Arrancar si no está 'running'
        start_list = []
        if controllers[start_controller_name] != "running":
            start_list.append(start_controller_name)

        # Parar todos los demás controladores que puedan causar conflicto, excepto franka_state_controller
        stop_list = []
        for ctrl_name, state in controllers.items():
            if (
                ctrl_name != start_controller_name
                and state == "running"
                and ctrl_name != "franka_state_controller"
            ):
                stop_list.append(ctrl_name)

        try:
            req = SwitchControllerRequest()
            req.start_controllers = start_list
            req.stop_controllers = stop_list
            req.strictness = SwitchControllerRequest.STRICT
            req.start_asap = True
            response = self.switch_controller_srv(req)
            if response.ok:
                rospy.loginfo(f"Controlador '{start_controller_name}' iniciado con éxito.")
                return True
            else:
                rospy.logerr(f"Fallo al iniciar el controlador '{start_controller_name}'.")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Llamada al servicio switch_controller fallida: {e}")
            return False

    def _get_loaded_controllers(self):
        """
        Devuelve la lista de controladores cargados en el controller_manager.
        """
        try:
            from controller_manager_msgs.srv import ListControllers
            rospy.wait_for_service('/controller_manager/list_controllers', timeout=5.0)
            list_srv = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
            resp = list_srv()
            return [c.name for c in resp.controller]
        except Exception as e:
            rospy.logerr(f"No se pudo obtener la lista de controladores: {e}")
            return []

    def execute_cb(self, goal):
        """
        Callback principal del servidor de acciones. Se llama cuando se recibe un nuevo goal.
        """
        rospy.loginfo("Nuevo Goal Recibido.")
        
        task_type = goal.task_type
        success = False

        if task_type == MoveFR3Goal.TASK_HOMING:
            rospy.loginfo("Tipo de Tarea: HOMING")
            success = self._handle_homing(goal)
            
        elif task_type == MoveFR3Goal.TASK_IMPEDANCE_MOVE:
            rospy.loginfo("Tipo de Tarea: IMPEDANCE MOVE")
            success = self._handle_impedance_move(goal)
        elif task_type == 2: # MoveFR3Goal.TASK_JOINT_POSE_EFFORT
            rospy.loginfo("Tipo de Tarea: JOINT POSE + ESFUERZO")
            success = self._handle_joint_pose_effort(goal)
        else:
            self._result.success = False
            self._result.message = f"Tipo de tarea desconocido: {task_type}"
            rospy.logerr(self._result.message)
            self._as.set_aborted(self._result)
            return

        # Finalizar la acción
        if success:
            self._result.success = True
            self._result.message = "Tarea completada con éxito."
            rospy.loginfo(self._result.message)
            self._as.set_succeeded(self._result)
        else:
            # Si se canceló, set_preempted() ya fue llamado
            if not self._as.is_preempt_requested():
                self._result.success = False
                self._result.message = "La tarea falló o fue abortada."
                rospy.logerr(self._result.message)
                self._as.set_aborted(self._result)
            elif self._as.is_preempt_requested():
                rospy.loginfo("La tarea fue preempted (cancelada).")


    def _handle_homing(self, goal):
        """
        Lógica para ejecutar la tarea de Homing usando el controlador de velocidad.
        Aplica limitación de error, velocidad y suavizado para evitar reflejos.
        Incluye velocidad angular.
        """
        if not self._switch_controller(self.VELOCITY_CONTROLLER_NAME):
            rospy.logerr("Homing fallido: No se pudo activar el controlador de velocidad.")
            return False
        
        rate = rospy.Rate(100) # 100 Hz
        twist_msg = Twist()

        # Pose objetivo del goal
        target_pos = np.array([
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.pose.position.z
        ])
        target_quat = np.array([
            goal.target_pose.pose.orientation.x,
            goal.target_pose.pose.orientation.y,
            goal.target_pose.pose.orientation.z,
            goal.target_pose.pose.orientation.w
        ])

        # Parámetros de suavizado
        alpha = 0.1  # Suavizado exponencial
        prev_vel_cmd = np.zeros(3)
        prev_ang_cmd = np.zeros(3)

        try:
            while not rospy.is_shutdown():
                
                # --- CANCELACIÓN DESDE EL BT (preempt) ---
                # if self._as.is_preempt_requested():
                #     rospy.loginfo("¡Acción de vel cancelada (preempted)! Iniciando rampa a cero.")
                #     # self._as.set_preempted()

                #     # Publicar inmediatamente comando de parada y luego intentar una deceleración suave
                #     rospy.loginfo("La tarea fue preempted (cancelada). Publicando parada segura.")

                if self._as.is_preempt_requested():
                    rospy.loginfo("¡Acción de Homing cancelada (preempted)! Iniciando rampa a cero.")
                    try:
                        self._ramp_down_velocity(rate_hz=100.0, alpha=0.9, timeout=3.0, tol=1e-3)
                    except Exception as e:
                        rospy.logwarn(f"Error durante deceleración tras preempt: {e}")

                    rospy.loginfo("Robot detenido tras preempt.")
                    self._as.set_preempted()
                    return False
                
                # --- Cálculo normal del comando de velocidad ---
                current_pos = np.array([
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                ])
                current_quat = np.array([
                    self.current_pose.pose.orientation.x,
                    self.current_pose.pose.orientation.y,
                    self.current_pose.pose.orientation.z,
                    self.current_pose.pose.orientation.w
                ])
                error_pos = target_pos - current_pos
                distance = np.linalg.norm(error_pos)

                # Error de orientación (en ángulo y eje)
                r_current = R.from_quat(current_quat)
                r_target = R.from_quat(target_quat)
                r_error = r_target * r_current.inv()
                angle = r_error.magnitude()
                axis = r_error.as_rotvec()
                # Limita el ángulo máximo para evitar saltos
                max_angle = 0.05  # radianes (~3 grados)
                if np.linalg.norm(axis) > max_angle:
                    axis = axis * (max_angle / np.linalg.norm(axis))

                # Control P para orientación
                ORIENTATION_P_GAIN = 0.8
                ang_cmd = ORIENTATION_P_GAIN * axis

                # Limita velocidad angular máxima
                MAX_ANGULAR_SPEED = 0.2  # rad/s
                ang_speed = np.linalg.norm(ang_cmd)
                if ang_speed > MAX_ANGULAR_SPEED:
                    ang_cmd = ang_cmd * (MAX_ANGULAR_SPEED / ang_speed)

                # Suavizado exponencial angular
                ang_cmd = alpha * ang_cmd + (1 - alpha) * prev_ang_cmd
                prev_ang_cmd = ang_cmd

                self._feedback.current_pose = self.current_pose
                self._feedback.distance_to_goal = distance
                self._as.publish_feedback(self._feedback)
                
                if distance < self.HOMING_POS_THRESHOLD and angle < 0.01:
                    rospy.loginfo("Homing completado. Iniciando rampa final a cero.")

                    try:
                        self._ramp_down_velocity(rate_hz=100.0, alpha=0.6, timeout=10.0, tol=1e-3)
                    except Exception as e:
                        rospy.logwarn(f"Error durante deceleración final de homing: {e}")

                    break

                # Limita el error máximo para evitar saltos
                max_error = 0.03  # 3 cm
                if np.linalg.norm(error_pos) > max_error:
                    error_pos = error_pos * (max_error / np.linalg.norm(error_pos))

                # Control P lineal
                vel_cmd = self.HOMING_P_GAIN * error_pos

                # Limita velocidad máxima lineal
                cmd_speed = np.linalg.norm(vel_cmd)
                if cmd_speed > self.HOMING_MAX_SPEED:
                    vel_cmd = vel_cmd * (self.HOMING_MAX_SPEED / cmd_speed)

                # Suavizado exponencial lineal
                vel_cmd = alpha * vel_cmd + (1 - alpha) * prev_vel_cmd
                prev_vel_cmd = vel_cmd

                twist_msg.linear.x = vel_cmd[0]
                twist_msg.linear.y = vel_cmd[1]
                twist_msg.linear.z = vel_cmd[2]
                twist_msg.angular.x = ang_cmd[0]
                twist_msg.angular.y = ang_cmd[1]
                twist_msg.angular.z = ang_cmd[2]

                twist_stamped_msg = TwistStamped()
                twist_stamped_msg.header.stamp = rospy.Time.now()
                twist_stamped_msg.twist = twist_msg

                self.velocity_cmd_pub.publish(twist_stamped_msg)
                rate.sleep()

        except Exception as e:
            rospy.logerr(f"Error durante el Homing: {e}")
            return False

        # Ojo: aquí ya hemos frenado suavemente, no mandes un cero de golpe
        rospy.loginfo("Homing finalizado con parada suave.")
        return True

    def _handle_impedance_move(self, goal):
        """
        Lógica corregida para ejecutar el movimiento de impedancia.
        """
        rospy.loginfo("--- Iniciando Movimiento de Impedancia ---")

        # 1. Cambiar al controlador de impedancia
        if not self._switch_controller(self.IMPEDANCE_CONTROLLER_NAME):
            rospy.logerr("Fallo al activar controlador de impedancia.")
            return False
        
        # 2. Preparar coordenadas
        target_pose = goal.target_pose
        
        # Validar espacio de trabajo
        if not self.esta_dentro_espacio_trabajo(target_pose):
            rospy.logerr("Target fuera del espacio de trabajo.")
            return False

        # Esperar pose válida inicial
        start_wait = rospy.Time.now()
        while self.current_pose.header.stamp == rospy.Time(0):
            if (rospy.Time.now() - start_wait).to_sec() > 2.0:
                rospy.logerr("Timeout esperando pose inicial.")
                return False
            rospy.sleep(0.1)

        # --- Preparación de la Trayectoria ---
        initial_pose = copy.deepcopy(self.current_pose)
        init_pos = np.array([initial_pose.pose.position.x, initial_pose.pose.position.y, initial_pose.pose.position.z])
        goal_pos = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z])

        # ### CAMBIO 1: Cálculo correcto de la duración basado en velocidad ###
        dist_total = np.linalg.norm(goal_pos - init_pos)
        velocidad = goal.max_velocity if goal.max_velocity > 0 else 0.1 # m/s por defecto
        
        # Tiempo = Distancia / Velocidad
        duration = dist_total / velocidad
        
        # Asegurar un tiempo mínimo para evitar movimientos explosivos (mínimo 2 segundos)
        duration = max(duration, 2.0)
        
        rospy.loginfo(f"Planificando: Distancia={dist_total:.3f}m, Velocidad={velocidad}m/s, Duración Calc={duration:.2f}s")

        # Configuración Splines
        frequency = 30  # Hz
        num_steps = int(duration * frequency)
        
        if num_steps < 2: 
            num_steps = 2 # Evitar errores con Spline si el paso es muy corto

        time_steps = np.linspace(0, 1, num_steps)
        
        # Interpolación Posición
        cubic_spline = scipy.interpolate.CubicSpline([0, 1], np.vstack([init_pos, goal_pos]), axis=0)
        trajectory_pos = cubic_spline(time_steps)
        
        # Interpolación Orientación
        init_quat_obj = R.from_quat([
            initial_pose.pose.orientation.x, initial_pose.pose.orientation.y, 
            initial_pose.pose.orientation.z, initial_pose.pose.orientation.w])
        goal_quat_obj = R.from_quat([
            target_pose.pose.orientation.x, target_pose.pose.orientation.y, 
            target_pose.pose.orientation.z, target_pose.pose.orientation.w])
        
        slerp = Slerp([0, 1], R.from_quat([init_quat_obj.as_quat(), goal_quat_obj.as_quat()]))
        trajectory_quat = slerp(time_steps)
        
        rate = rospy.Rate(frequency)
        threshold = 0.03 # 3 cm

        # --- Ejecución de la Trayectoria ---
        rospy.loginfo("Ejecutando trayectoria...")
        try:
            for i in range(num_steps):
                if self._as.is_preempt_requested():
                    rospy.loginfo("Cancelado durante trayectoria.")
                    self._as.set_preempted()
                    return False

                # Construir mensaje
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "fr3_link0" # Asegúrate que este es el frame correcto
                
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = trajectory_pos[i]
                qx, qy, qz, qw = trajectory_quat[i].as_quat()
                pose_msg.pose.orientation.x = qx
                pose_msg.pose.orientation.y = qy
                pose_msg.pose.orientation.z = qz
                pose_msg.pose.orientation.w = qw
                
                self.impedance_pose_pub.publish(pose_msg)
                
                # Feedback
                current_pos_array = np.array([
                    self.current_pose.pose.position.x, 
                    self.current_pose.pose.position.y, 
                    self.current_pose.pose.position.z
                ])
                distance_error = np.linalg.norm(goal_pos - current_pos_array)
                
                self._feedback.current_pose = self.current_pose
                self._feedback.distance_to_goal = distance_error
                self._as.publish_feedback(self._feedback)
                
                rate.sleep()
            
            # Asegurarse de publicar el último punto exacto al final
            self.impedance_pose_pub.publish(target_pose)

        except Exception as e:
            rospy.logerr(f"Excepción en bucle: {e}")
            return False

        # ### CAMBIO 2: Fase de Asentamiento (Settling Phase) ###
        # El robot necesita tiempo físico para llegar al último punto enviado.
        rospy.loginfo("Trayectoria enviada. Esperando convergencia física (settling)...")
        
        settling_timeout = 3.0 # Esperar máximo 3 segundos extra
        settling_start = rospy.Time.now()
        
        while (rospy.Time.now() - settling_start).to_sec() < settling_timeout:
            if self._as.is_preempt_requested():
                self._as.set_preempted()
                return False

            # Seguir publicando el objetivo final para mantener la fuerza elástica
            target_pose.header.stamp = rospy.Time.now()
            self.impedance_pose_pub.publish(target_pose)

            current_pos_array = np.array([
                self.current_pose.pose.position.x, 
                self.current_pose.pose.position.y, 
                self.current_pose.pose.position.z
            ])
            final_error = np.linalg.norm(goal_pos - current_pos_array)
            
            if final_error < threshold:
                rospy.loginfo(f"¡Objetivo alcanzado! Error final: {final_error:.4f} m")
                return True
            
            rate.sleep()

        # Verificación final tras timeout
        if final_error < threshold:
             return True
        else:
             rospy.logwarn(f"Tiempo de espera agotado. El robot no convergió. Error final: {final_error:.4f} m")
             return False

    def _handle_joint_pose_effort(self, goal):
        """
        Cambia al controlador de esfuerzo y publica la trayectoria de joint pose.
        """
        # Cambia al controlador de esfuerzo
        if not self._switch_controller(self.EFFORT_CONTROLLER_NAME):
            rospy.logerr("No se pudo activar el controlador de esfuerzo.")
            return False

        # Prepara el mensaje de trayectoria
        from control_msgs.msg import FollowJointTrajectoryActionGoal
        from trajectory_msgs.msg import JointTrajectoryPoint

        traj_goal = FollowJointTrajectoryActionGoal()
        traj_goal.goal.trajectory.joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        point = JointTrajectoryPoint()
        point.positions = [0, -0.785, 0, -2.355, 0, 1.571, 0.785]
        point.velocities = [0, 0, 0, 0, 0, 0, 0]
        point.time_from_start = rospy.Duration(5.0) # TODO: Revisar si quiero especificar la duración en el mensaje. Dilema: velocidad vs tiempo
        traj_goal.goal.trajectory.points.append(point)
        traj_goal.goal.goal_time_tolerance = rospy.Duration(1.0)

        # Publica el goal en el topic del controlador
        pub = rospy.Publisher(
            "/effort_joint_trajectory_controller/follow_joint_trajectory/goal",
            FollowJointTrajectoryActionGoal,
            queue_size=1
        )
        rospy.sleep(0.5)  # Espera a que el publisher se conecte
        pub.publish(traj_goal)
        rospy.loginfo("Goal de joint pose publicado al controlador de esfuerzo.")

        # Espera a que termine el movimiento y actualiza el feedback
        timeout = 10.0  # segundos
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < timeout:
            if self.effort_status is not None:
                rospy.loginfo_throttle_identical(10,f"Status actual del goal: {self.effort_status}") # Periodo alto para que solo mande un único mensaje identico
                # Puedes retornar el valor directamente o usarlo en la lógica
                if self.effort_status == 3:  # SUCCEEDED
                    rospy.loginfo("Trayectoria ejecutada correctamente (status SUCCEEDED).")
                    return True
                elif self.effort_status in [4, 5, 8, 9]:  # ABORTED, REJECTED, RECALLED, LOST
                    rospy.logwarn(f"Trayectoria abortada o rechazada (status={self.effort_status}).")
                    return False
            rospy.sleep(0.1)
        rospy.logwarn("Timeout esperando status del controlador de esfuerzo.")
        return False

    def esta_dentro_espacio_trabajo(self, pose):
        """
        Verifica si una pose está dentro del espacio de trabajo definido.
        """
        x_limits = (-0.35, 0.90)
        y_limits = (-0.80, 0.80)
        z_limits = (0.10, 0.90)

        x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
        return (x_limits[0] <= x <= x_limits[1] and
                y_limits[0] <= y <= y_limits[1] and
                z_limits[0] <= z <= z_limits[1])

if __name__ == "__main__":
    try:
        server = Fr3ActionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass