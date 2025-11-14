#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación del BT de agarre de muñeca con py_trees_ros.
Estructura: Try-Catch-Finally robusta.
Stack: ROS 1 Noetic, Python, py_trees, py_trees_ros
"""

import py_trees
import py_trees_ros
import rospy
import time
import std_msgs.msg
import franka_msgs.msg
from skeleton_3d.msg import Skeleton3D
from geometry_msgs.msg import PoseStamped
import numpy as np
from gripper_4f.srv import SetPWM, SetPWMResponse
import math
from scipy.spatial.transform import Rotation as R
import actionlib
from franka_concerto.msg import MoveFR3Action, MoveFR3Goal
from actionlib_msgs.msg import GoalStatus
from franka_concerto.funciones_utiles import calculate_gripper_position_forearm_correction, calculate_quaternion_0_F

# --- Nodos Hoja (Plantillas) ---
# Rellena la lógica de ROS (subs, pubs, services, actions) aquí.
class SubscribeDummyTopic(py_trees.behaviour.Behaviour):
    """Suscribe a /dummy_topic (std_msgs/Bool) y lo pone en el blackboard."""
    def __init__(self, name="SubscribeDummyTopic"):
        super(SubscribeDummyTopic, self).__init__(name)
        self.topic_name = "/dummy_topic"
        self.msg = None
        self.subscriber = None

    def setup(self, timeout):
        self.subscriber = rospy.Subscriber(self.topic_name, std_msgs.msg.Bool, self.callback)
        return True

    def callback(self, msg):
        self.msg = msg

    def update(self):
        if self.msg is not None:
            py_trees.blackboard.Blackboard().set("dummy_topic_value", self.msg.data)
            rospy.loginfo(f"Dummy topic value set on blackboard: {self.msg.data}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class SubscribeRobotMode(py_trees.behaviour.Behaviour):
    """Suscribe a /franka_states y extrae solo robot_mode."""
    def __init__(self, name="SubscribeRobotMode"):
        super(SubscribeRobotMode, self).__init__(name)
        self.topic_name = "/franka_state_controller/franka_states"
        self.robot_mode = None
        self.subscriber = None

    def setup(self, timeout):
        self.subscriber = rospy.Subscriber(self.topic_name, franka_msgs.msg.FrankaState, self.callback)
        return True

    def callback(self, msg):
        self.robot_mode = msg.robot_mode  # Extrae solo el campo que te interesa

    def update(self):
        if self.robot_mode is not None:
            py_trees.blackboard.Blackboard().set("robot_mode", self.robot_mode)
            rospy.loginfo(f"Robot mode set on blackboard: {self.robot_mode}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING  
    
class SubscribeSkeleton3D(py_trees.behaviour.Behaviour):
    """Suscribe a /skeleton_3d y guarda el último mensaje en el blackboard."""
    def __init__(self, name="SubscribeSkeleton3D"):
        super(SubscribeSkeleton3D, self).__init__(name)
        self.topic_name = "/skeleton_3D"
        self.rwrist = None
        self.relbow = None
        self.rshoulder = None
        self.subscriber = None

    def setup(self, timeout):
        self.subscriber = rospy.Subscriber(self.topic_name, Skeleton3D, self.callback)
        return True

    def callback(self, msg):
        # Extrae los keypoints relevantes
        self.rwrist = msg.keypoints[10]
        self.relbow = msg.keypoints[8]
        self.rshoulder = msg.keypoints[6]
        # Calcular el vector normal al plano formado por los 3 puntos
        p1 = np.array([self.rshoulder.x, self.rshoulder.y, self.rshoulder.z])
        p2 = np.array([self.relbow.x, self.relbow.y, self.relbow.z])
        p3 = np.array([self.rwrist.x, self.rwrist.y, self.rwrist.z])
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
        self.normal = normal

    def update(self):
        if self.rshoulder is not None and self.relbow is not None and self.rwrist is not None and hasattr(self, "normal"):
            py_trees.blackboard.Blackboard().set("rshoulder_kp", self.rshoulder)
            py_trees.blackboard.Blackboard().set("relbow_kp", self.relbow)
            py_trees.blackboard.Blackboard().set("rwrist_kp", self.rwrist)
            py_trees.blackboard.Blackboard().set("wrist_normal", self.normal)
            rospy.loginfo("Skeleton3D keypoints and normal set on blackboard.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING
    

class SubscribeGraspState(py_trees.behaviour.Behaviour):
    """Suscribe a /gripper_4f/grasp_state (std_msgs/Bool) y guarda el bool en el blackboard."""
    def __init__(self, name="SubscribeGraspState"):
        super(SubscribeGraspState, self).__init__(name)
        self.topic_name = "/gripper_4f/grasp_state"
        self.state = None
        self.subscriber = None

    def setup(self, timeout):
        self.subscriber = rospy.Subscriber(self.topic_name, std_msgs.msg.Bool, self.callback)
        return True

    def callback(self, msg):
        self.state = bool(msg.data)

    def update(self):
        if self.state is not None:
            py_trees.blackboard.Blackboard().set("gripper_closed", self.state)
            rospy.loginfo(f"Grasp state set on blackboard: {self.state}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class SubscribeEEPose(py_trees.behaviour.Behaviour):
    """Subscribe to /franka_state_controller/ee_pose (geometry_msgs/PoseStamped)
    and store the latest PoseStamped on the blackboard as 'ee_pose'.
    """
    def __init__(self, name="SubscribeEEPose"):
        super(SubscribeEEPose, self).__init__(name)
        self.topic_name = "/franka_state_controller/ee_pose"
        self.msg = None
        self.subscriber = None

    def setup(self, timeout):
        self.subscriber = rospy.Subscriber(self.topic_name, PoseStamped, self.callback)
        return True

    def callback(self, msg): 
        self.msg = msg

    def update(self):
        if self.msg is not None:
            py_trees.blackboard.Blackboard().set("ee_pose", self.msg)
            rospy.loginfo("EE pose set on blackboard.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


# --- Hojas de Pre-checks ---

class RobotModeOk(py_trees.behaviour.Behaviour):
    """(Condición) Robot_mode ok?"""
    def __init__(self, name="RobotModeOk"):
        super(RobotModeOk, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.mode_required_list = [1, 2, 3, 4]  # Modo de ejecución [luz verde = 1]
    
    def update(self):
        rospy.loginfo("Check: Robot Mode OK?")
        self.robot_mode = self.blackboard.get("robot_mode")

        if self.robot_mode in self.mode_required_list:
            rospy.loginfo(f"[{self.name}] Check: Robot Mode OK? -> YES")
            return py_trees.common.Status.SUCCESS
        else:
            rospy.logerr(f"[{self.name}] Robot Mode OK? -> NO (Current: {self.robot_mode})")
            return py_trees.common.Status.FAILURE

class IsGripperOpen(py_trees.behaviour.Behaviour):
    """(Condición) Garra abierta?"""
    def __init__(self, name="IsGripperOpen"):
        super(IsGripperOpen, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        rospy.loginfo("Check: Gripper Open?")
        # TODO: Comprobar el estado de la garra
        # Simulación: Falla la primera vez para forzar la acción
        
        if self.blackboard.get("gripper_closed") == True: # garra cerrada
            rospy.loginfo("Check: Gripper Open? -> No")
            return py_trees.common.Status.FAILURE
        
        rospy.loginfo("Check: Gripper Open? -> YES")
        return py_trees.common.Status.SUCCESS

class OpenGripper(py_trees.behaviour.Behaviour):
    """(Acción) Abrir garra"""
    def __init__(self, name="OpenGripper"):
        super(OpenGripper, self).__init__(name)
        
        self.pwm_to_open = -100  # Valor de PWM para abrir la garra
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        rospy.wait_for_service('/gripper_4f/set_pwm')
        self.set_pwm_service = rospy.ServiceProxy('/gripper_4f/set_pwm', SetPWM)
        return super().setup(timeout)

    def initialise(self):
        
        try:
            response = self.set_pwm_service(self.pwm_to_open) # Llamar al servicio 

            if response.success:
                self.service_call_succeeded = True
                rospy.loginfo(f"[{self.name}] Gripper Open Command Sent!")
            else:
                self.service_call_succeeded = False
                rospy.logerr(f"[{self.name}] Gripper Open Command Failed!")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        
    def update(self):

        # Si la llamada en initialise() falló, fallamos inmediatamente.
        if not self.service_call_succeeded:
            return py_trees.common.Status.FAILURE
        
        gripper_closed_status = self.blackboard.get("gripper_closed")

        # Comprobando el estado de la garra
        if gripper_closed_status is False: # garra abierta
            rospy.loginfo(f"[{self.name}] Gripper is now Open!")
            return py_trees.common.Status.SUCCESS
        elif gripper_closed_status is True: # garra cerrada
            return py_trees.common.Status.RUNNING
        else: # None. blackboard sin datos
            rospy.logdebug(f"[{self.name}] Esperando datos en blackboard 'gripper_closed'...")
            return py_trees.common.Status.RUNNING

class IsAtHomePose(py_trees.behaviour.Behaviour):
    """(Condición) Robot pose inicial?"""
    def __init__(self, name="IsAtHomePose", pos_tol=0.02, ori_tol_deg=5.0):
        super(IsAtHomePose, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        # homing_pose ya la estás definiendo en __init__ (ej. como PoseStamped)
        self.pos_tol = float(pos_tol)                # tolerancia de posición en metros
        self.ori_tol_rad = math.radians(ori_tol_deg) # tolerancia de orientación en radianes

        # Definir la pose de homing
        self.homing_pose = PoseStamped()
        self.homing_pose.header.frame_id = "fr3_link0"
        self.homing_pose.pose.position.x = 0.31
        self.homing_pose.pose.position.y = 0.0
        self.homing_pose.pose.position.z = 0.47
        self.homing_pose.pose.orientation.x = 1.0
        self.homing_pose.pose.orientation.y = 0.0
        self.homing_pose.pose.orientation.z = 0.0
        self.homing_pose.pose.orientation.w = 0.0

    def _quat_angle_diff(self, q1, q2):
        """
        Devuelve el ángulo mínimo entre dos quaternions (rad).
        q1, q2: iterables (x,y,z,w)
        """
        r1 = R.from_quat([q1[0], q1[1], q1[2], q1[3]])
        r2 = R.from_quat([q2[0], q2[1], q2[2], q2[3]])
        # r_rel es la rotación que lleva r1 a r2
        r_rel = r2 * r1.inv()
        angle = r_rel.magnitude()  # ángulo en rad
        # normalizar al intervalo [0, pi]
        return abs((angle + math.pi) % (2*math.pi) - math.pi)

    def update(self):
        rospy.logdebug("Check: At Home Pose? (tolerant)")
        ee_pose = self.blackboard.get("ee_pose")
        if ee_pose is None:
            rospy.loginfo("Check: At Home Pose? -> NO (no ee_pose on blackboard)")
            return py_trees.common.Status.FAILURE

        # opcional: comprobar frames
        try:
            if hasattr(ee_pose, 'header') and hasattr(self.homing_pose, 'header'):
                if ee_pose.header.frame_id != self.homing_pose.header.frame_id:
                    rospy.logwarn_throttle(30, f"EE pose frame '{ee_pose.header.frame_id}' != homing_pose frame '{self.homing_pose.header.frame_id}'")
        except Exception:
            pass

        # Distancia euclídea
        dx = ee_pose.pose.position.x - self.homing_pose.pose.position.x
        dy = ee_pose.pose.position.y - self.homing_pose.pose.position.y
        dz = ee_pose.pose.position.z - self.homing_pose.pose.position.z
        pos_err = (dx*dx + dy*dy + dz*dz)**0.5

        # Diferencia angular entre quaternions
        q_ee = (ee_pose.pose.orientation.x, ee_pose.pose.orientation.y, ee_pose.pose.orientation.z, ee_pose.pose.orientation.w)
        q_ref = (self.homing_pose.pose.orientation.x, self.homing_pose.pose.orientation.y, self.homing_pose.pose.orientation.z, self.homing_pose.pose.orientation.w)
        ori_err = self._quat_angle_diff(q_ref, q_ee)

        rospy.loginfo(f"AtHome check: pos_err={pos_err:.4f} m, ori_err={math.degrees(ori_err):.2f} deg (tol pos={self.pos_tol} m, tol ori={math.degrees(self.ori_tol_rad):.2f} deg)")

        if pos_err <= self.pos_tol and ori_err <= self.ori_tol_rad:
            rospy.loginfo("Check: At Home Pose? -> YES")
            return py_trees.common.Status.SUCCESS
        else:
            rospy.loginfo("Check: At Home Pose? -> NO")
            return py_trees.common.Status.FAILURE

class GoToHome(py_trees.behaviour.Behaviour):
    """(Acción) Plan y ejecutar al reposo usando fr3_motion_server action."""
    def __init__(self, name="GoToHome"):
        super(GoToHome, self).__init__(name)
        self.done = False
        self._client = None
        self._goal_sent = False
        self._succeeded = False
        self._server_available = False

    def setup(self, timeout):
        # Crear cliente de acción y esperar al servidor (no bloqueante excesivo)
        self._client = actionlib.SimpleActionClient('fr3_motion_server', MoveFR3Action)
        try:
            ok = self._client.wait_for_server(rospy.Duration(2.0))
            if ok:
                self._server_available = True
                rospy.loginfo(f"[{self.name}] Conectado a fr3_motion_server.")
            else:
                self._server_available = False
                rospy.logwarn(f"[{self.name}] No se encontró fr3_motion_server (timeout).")
        except Exception as e:
            self._server_available = False
            rospy.logwarn(f"[{self.name}] Excepción esperando al servidor: {e}")
        return True

    def initialise(self):
        self.done = False
        self._goal_sent = False
        self._succeeded = False

        if not self._server_available:
            rospy.logerr(f"[{self.name}] Servidor de acción no disponible.")
            # No abortamos el árbol aquí, devolvemos FAILURE en update()
            return

        # Construir el goal (igual que el ejemplo de rostopic)
        goal = MoveFR3Goal()
        goal.task_type = 2  # como en tu ejemplo (ajusta si prefieres 0=HOMING)
        # Rellenar target_pose (ejemplo: todos a cero; cambia frame si quieres)
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = ""
        goal.target_pose.pose.position.x = 0.0
        goal.target_pose.pose.position.y = 0.0
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = 0.0
        goal.target_pose.pose.orientation.w = 1.0
        goal.max_velocity = 0.0

        # Enviar goal y registrar callbacks
        try:
            self._client.send_goal(goal, done_cb=self._done_cb, feedback_cb=self._feedback_cb)
            self._goal_sent = True
            rospy.loginfo(f"[{self.name}] Goal enviado a fr3_motion_server.")
        except Exception as e:
            rospy.logerr(f"[{self.name}] Error enviando goal: {e}")
            self._goal_sent = False
            self._succeeded = False
            self.done = True

    def _feedback_cb(self, feedback):
        # Opcional: puedes publicar feedback en el blackboard si lo deseas
        pass

    def _done_cb(self, state, result):
        # state es un entero (GoalStatus); 3 == SUCCEEDED
        try:
            if state == GoalStatus.SUCCEEDED:
                self._succeeded = True
                rospy.loginfo(f"[{self.name}] Acción completada: SUCCEEDED")
            else:
                self._succeeded = False
                rospy.logwarn(f"[{self.name}] Acción terminada con estado {state}")
        except Exception:
            self._succeeded = False
        finally:
            self.done = True

    def update(self):
        if not self._server_available:
            # Servidor ausente => fallo de la acción
            rospy.logerr(f"[{self.name}] update: servidor no disponible.")
            return py_trees.common.Status.FAILURE

        # Goal enviado y aún no finalizado
        if self._goal_sent and not self.done:
            return py_trees.common.Status.RUNNING

        # Si finalizó, devolver SUCCESS/FAILURE según el resultado
        if self.done:
            if self._succeeded:
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE

        # Caso por defecto: iniciar (si initialise no se ejecutó) -> RUNNING
        return py_trees.common.Status.RUNNING

# --- Hojas de Tarea Principal ---

class IsWristDataAvailable(py_trees.behaviour.Behaviour):
    """(Condición) Muñeca y vector normal disponibles"""
    def __init__(self, name="IsWristDataAvailable"):
        super(IsWristDataAvailable, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        rospy.loginfo("Check: Wrist Data Available?")
        rwrist = self.blackboard.get("rwrist_kp")
        normal = self.blackboard.get("wrist_normal")
        
        if rwrist is None or normal is None:
            return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.SUCCESS

class IsTrajectoryFree(py_trees.behaviour.Behaviour):
    """(Condición) Trayectoria libre"""
    def __init__(self, name="IsTrajectoryFree"):
        super(IsTrajectoryFree, self).__init__(name)
    
    def update(self):
        rospy.loginfo("Check: Trajectory Free?")
        # TODO: Usar MoveIt para comprobar la validez de la escena
        return py_trees.common.Status.SUCCESS

class PlanAndApproach(py_trees.behaviour.Behaviour):
    """(Acción) Plan y aprox — calcula approach = kp_wrist + normal*0.15 y envía ese pose como goal (IMPEDANCE_MOVE)."""
    def __init__(self, name="PlanAndApproach"):
        super(PlanAndApproach, self).__init__(name)
        self.done = False
        self._client = None
        self._goal_sent = False
        self._succeeded = False
        self._server_available = False
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        # preparar cliente de acción (espera corta)
        self._client = actionlib.SimpleActionClient('fr3_motion_server', MoveFR3Action)
        try:
            ok = self._client.wait_for_server(rospy.Duration(2.0))
            if ok:
                self._server_available = True
                rospy.loginfo(f"[{self.name}] Conectado a fr3_motion_server.")
            else:
                self._server_available = False
                rospy.logwarn(f"[{self.name}] No se encontró fr3_motion_server (timeout).")
        except Exception as e:
            self._server_available = False
            rospy.logwarn(f"[{self.name}] Excepción esperando al servidor: {e}")
        return True

    def initialise(self):
        self.done = False
        self._goal_sent = False
        self._succeeded = False

        # Leer datos del blackboard
        rwrist = self.blackboard.get("rwrist_kp")
        relbow = self.blackboard.get("relbow_kp")
        normal = self.blackboard.get("wrist_normal")

        if rwrist is None or relbow is None or normal is None:
            rospy.logerr(f"[{self.name}] Faltan datos en blackboard (rwrist/relbow/wrist_normal).")
            return

        # Calcular vectores np
        try:
            rw = np.array([rwrist.x, rwrist.y, rwrist.z])
            re = np.array([relbow.x, relbow.y, relbow.z])
            forearm = rw - re
            forearm_norm = np.linalg.norm(forearm)
            if forearm_norm > 1e-8:
                forearm = forearm / forearm_norm
            else:
                rospy.logwarn(f"[{self.name}] forearm vector norma casi cero. Usando fallback (0,0,1).")
                forearm = np.array([0.0, 0.0, 1.0])

            normal_arr = np.array(normal)
            norm_norm = np.linalg.norm(normal_arr)
            if norm_norm <= 1e-8:
                rospy.logwarn(f"[{self.name}] normal vector norma casi cero. Usando fallback (0,0,1).")
                normal_arr = np.array([0.0, 0.0, 1.0])
            else:
                normal_arr = normal_arr / norm_norm

        except Exception as e:
            rospy.logerr(f"[{self.name}] Error preparando vectores: {e}")
            return

        # calcular pose de aproximación (kp_wrist + normal * 0.15) con corrección de antebrazo (-0.1)
        try:
            approach_point = calculate_gripper_position_forearm_correction(
                rw, normal_arr, forearm, 0.15, -0.1
            )
        except Exception as e:
            rospy.logerr(f"[{self.name}] Error calculando posición de approach: {e}")
            return

        # calcular orientación (igual que hri_states_machine)
        try:
            # calculate_quaternion_0_F espera vectores; en hri_states_machine se pasa (-forearm, -normal)
            quat_pose = calculate_quaternion_0_F(-forearm, -normal_arr)
            qx = quat_pose.x
            qy = quat_pose.y
            qz = quat_pose.z
            qw = quat_pose.w
        except Exception as e:
            rospy.logerr(f"[{self.name}] Error calculando orientación: {e}")
            return

        rospy.loginfo(f"[{self.name}] Approach pose: x={approach_point.x:.3f}, y={approach_point.y:.3f}, z={approach_point.z:.3f}")

        if not self._server_available:
            rospy.logerr(f"[{self.name}] Servidor de acción no disponible. No se enviará goal.")
            return

        # Construir y enviar goal con la pose calculada (IMPEDANCE_MOVE)
        goal = MoveFR3Goal()
        goal.task_type = 1  # TASK_IMPEDANCE_MOVE
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "fr3_link0"
        goal.target_pose.pose.position.x = approach_point.x
        goal.target_pose.pose.position.y = approach_point.y
        goal.target_pose.pose.position.z = approach_point.z
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        goal.max_velocity = 0.1

        try:
            self._client.send_goal(goal, done_cb=self._done_cb, feedback_cb=self._feedback_cb)
            self._goal_sent = True
            rospy.logwarn(f"[{self.name}] !!!Goal (approach pose) enviado a fr3_motion_server.")
        except Exception as e:
            rospy.logerr(f"[{self.name}] Error enviando goal: {e}")
            self._goal_sent = False
            self._succeeded = False
            self.done = True

    def _feedback_cb(self, feedback):
        # opcional: guardar feedback en blackboard
        pass

    def _done_cb(self, state, result):
        try:
            if state == GoalStatus.SUCCEEDED:
                self._succeeded = True
                rospy.logwarn(f"[{self.name}] Acción completada: SUCCEEDED")
            else:
                self._succeeded = False
                rospy.logwarn(f"[{self.name}] Acción terminada con estado {state}")
        except Exception:
            self._succeeded = False
        finally:
            self.done = True

    def update(self):
        if not self._server_available:
            rospy.logwarn("DEBUG: server not available in update()")
            return py_trees.common.Status.FAILURE

        # Si tenemos cliente, consultar su estado y devolver RUNNING si hay un goal activo/pending
        try:
            if self._client is not None:
                state = self._client.get_state()
                if state in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                    rospy.logdebug(f"[{self.name}] Action state {state} -> RUNNING")
                    return py_trees.common.Status.RUNNING
                # Si el estado es terminal y todavía no hemos recibido el callback, lo reflejamos:
                if state == GoalStatus.SUCCEEDED:
                    self._succeeded = True
                    self.done = True
                elif state in (GoalStatus.ABORTED, GoalStatus.PREEMPTED, GoalStatus.REJECTED,
                               GoalStatus.RECALLED, GoalStatus.LOST):
                    self._succeeded = False
                    self.done = True
        except Exception as e:
            rospy.logwarn(f"[{self.name}] No se pudo consultar estado del action client: {e}")

        # Si enviamos un goal pero aún no ha terminado (fallback a banderas internas)
        if self._goal_sent and not self.done:
            rospy.logdebug("DEBUG: goal sent and not done yet (internal flag)")
            return py_trees.common.Status.RUNNING

        # Si terminó, devolver SUCCESS/FAILURE según el resultado
        if self.done:
            rospy.logdebug("DEBUG: done -> returning final status")
            return py_trees.common.Status.SUCCESS if self._succeeded else py_trees.common.Status.FAILURE

        # Caso por defecto: RUNNING (evita reentradas que provoquen initialise/reenvío)
        rospy.logdebug("DEBUG: default running")
        return py_trees.common.Status.RUNNING

class IsWristStable(py_trees.behaviour.Behaviour):
    """(Condición) Muñeca aproxima (estable)"""
    def __init__(self, name="IsWristStable"):
        super(IsWristStable, self).__init__(name)
    
    def update(self):
        rospy.loginfo("Check: Wrist Stable?")
        # TODO: Comprobar que la pose de la muñeca no ha cambiado
        return py_trees.common.Status.SUCCESS

class PlanAndContact(py_trees.behaviour.Behaviour):
    """(Acción) Plan y contacto"""
    def __init__(self, name="PlanAndContact"):
        super(PlanAndContact, self).__init__(name)
        self.done = False

    def initialise(self):
        self.done = False
        rospy.loginfo("Action: Planning and Executing Contact...")
        # TODO: Usar controlador de impedancia o movimiento de contacto

    def update(self):
        if not self.done:
            self.done = True
            rospy.loginfo("Action: Contact Made!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class CloseGripper(py_trees.behaviour.Behaviour):
    """(Acción) Cerrar garra"""
    def __init__(self, name="CloseGripper"):
        super(CloseGripper, self).__init__(name)
        self.done = False

    def initialise(self):
        self.done = False
        rospy.loginfo("Action: Closing Gripper...")
        # TODO: Enviar comando de agarre (franka_gripper/grasp)
        # Importante: resetear la condición 'IsGripperOpen' para el cleanup
        setattr(IsGripperOpen, "been_opened", False)

    def update(self):
        if not self.done:
            self.done = True
            rospy.loginfo("Action: Gripper Closed!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class IsGraspGood(py_trees.behaviour.Behaviour):
    """(Condición) garra cerrada en antebrazo?"""
    def __init__(self, name="IsGraspGood"):
        super(IsGraspGood, self).__init__(name)
    
    def update(self):
        rospy.loginfo("Check: Good Grasp?")
        # TODO: Comprobar el ancho de la garra (franka_gripper/state)
        # --- ¡Descomenta esto para probar el fallo! ---
        # rospy.logerr("Check: Grasp FAILED!")
        # return py_trees.common.Status.FAILURE
        # ---
        rospy.loginfo("Check: Grasp OK!")
        return py_trees.common.Status.SUCCESS

class RunNeuralNet(py_trees.behaviour.Behaviour):
    """(Acción) Red neuronal"""
    def __init__(self, name="RunNeuralNet"):
        super(RunNeuralNet, self).__init__(name)
        self.done = False

    def initialise(self):
        self.done = False
        rospy.loginfo("Action: Running Neural Net...")
        # TODO: Llamar al servicio de la NN

    def update(self):
        if not self.done:
            self.done = True
            rospy.loginfo("Action: Neural Net Finished!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

# --- Hojas de Clean Up ---

class IsAtRetractPose(py_trees.behaviour.Behaviour):
    """(Condición) Aproximado? (En pose de retirada)"""
    def __init__(self, name="IsAtRetractPose"):
        super(IsAtRetractPose, self).__init__(name)
    
    def update(self_):
        rospy.loginfo("Check: Already Retracted?")
        # TODO: Comprobar si ya estamos en una pose segura de retirada
        return py_trees.common.Status.FAILURE # Forzar la acción de retirada

class Retract(py_trees.behaviour.Behaviour):
    """(Acción) Retirar"""
    def __init__(self, name="Retract"):
        super(Retract, self).__init__(name)
        self.done = False

    def initialise(self):
        self.done = False
        rospy.loginfo("Action: Retracting from user...")
        # TODO: Enviar goal de movimiento relativo (ej. -10cm en X del end-effector)

    def update(self):
        if not self.done:
            self.done = True
            rospy.loginfo("Action: Retract Complete!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


# --- Funciones de Construcción ---

def create_pre_checks_subtree():
    """Crea el sub-árbol de Pre-checks."""
    
    # ? (Selector) "Asegurar Garra Abierta"
    ensure_gripper_open = py_trees.composites.Selector(
        name="Asegurar Garra Abierta",
        memory=False,
        children=[
            IsGripperOpen(),
            OpenGripper()
        ]
    )

    # ? (Selector) "Asegurar Pose Inicial"
    ensure_home_pose = py_trees.composites.Selector(
        name="Asegurar Pose Inicial",
        memory=True,
        children=[
            IsAtHomePose(),
            GoToHome()
        ]
    )

    # -> (Sequence) "Pre-checks"
    pre_checks_root = py_trees.composites.Sequence(
        name="Pre-checks",
        memory=True,
        children=[
            ensure_gripper_open,
            RobotModeOk(),
            # ensure_home_pose # TODO: Revisar porq no se puede mover durante el movimiento porq se pelea con esta condicion
        ]
    )
    return pre_checks_root

def create_main_task_subtree():
    """Crea el sub-árbol de Tarea Principal."""
    
    # -> (Sequence) "Aproximación"
    approach_seq = py_trees.composites.Sequence(
        name="Aproximación",
        memory=True,
        children=[
            IsTrajectoryFree(),
            PlanAndApproach()
        ]
    )
    
    # -> (Sequence) "Contacto"
    contact_seq = py_trees.composites.Sequence(
        name="Contacto",
        memory=True,
        children=[
            IsWristStable(),
            PlanAndContact()
        ]
    )

    # -> (Sequence) "Agarre"
    grasp_seq = py_trees.composites.Sequence(
        name="Agarre",
        memory=True,
        children=[
            CloseGripper(),
            IsGraspGood(),
            RunNeuralNet()
        ]
    )

    # -> (Sequence) "Tarea principal"
    main_task_root = py_trees.composites.Sequence(
        name="Tarea principal",
        memory=True,
        children=[
            IsWristDataAvailable(),
            approach_seq,
            contact_seq,
            grasp_seq
        ]
    )
    return main_task_root

def create_cleanup_subtree():
    """Crea el sub-árbol de Clean up (robusto)."""

    # ? (Selector) "Asegurar Garra Abierta"
    ensure_gripper_open = py_trees.composites.Selector(
        name="Asegurar Garra Abierta (CU)",
        memory=True,
        children=[
            IsGripperOpen(),
            OpenGripper()
        ]
    )

    # ? (Selector) "Asegurar Retirada"
    ensure_retract = py_trees.composites.Selector(
        name="Asegurar Retirada (CU)",
        memory=True,
        children=[
            IsAtRetractPose(),
            Retract()
        ]
    )

    # ? (Selector) "Asegurar Reposo"
    ensure_home = py_trees.composites.Selector(
        name="Asegurar Reposo (CU)",
        memory=True,
        children=[
            IsAtHomePose(),
            GoToHome()
        ]
    )

    # -> (Sequence) "Clean up"
    cleanup_root = py_trees.composites.Sequence(
        name="Clean up",
        memory=True,
        children=[
            ensure_gripper_open,
            ensure_retract,
            ensure_home
        ]
    )
    return cleanup_root


def create_root():
    """
    Construye el árbol de comportamiento raíz completo.
    Estructura: Try-Catch-Finally
    """
    
    # Construir los sub-árboles
    pre_checks_subtree = create_pre_checks_subtree()
    main_task_subtree = create_main_task_subtree()
    cleanup_subtree = create_cleanup_subtree()

    # -> (Sequence) "Pre-checks + Tarea principal"
    full_task_sequence = py_trees.composites.Sequence(
        name="Attempt block",
        memory=True, # Memoria para la secuencia de tareas
        children=[
            pre_checks_subtree,
            main_task_subtree
        ]
    )

    # Nodo que siempre devuelve SUCCESS, usado como "catch"
    always_success_node = py_trees.behaviours.Success(name="Always success")

    # ? (Selector) "Try-catch structure"
    try_catch_block = py_trees.composites.Selector(
        name="Try-catch structure",
        memory=False, # Sin memoria, para que siempre re-intente la tarea
        children=[
            full_task_sequence,
            always_success_node
        ]
    )

    # -> (Sequence) "Try-finally structure" (RAÍZ)
    root = py_trees.composites.Sequence(
        name="Try-finally structure",
        memory=True,
        children=[
            try_catch_block,
            # cleanup_subtree # DEBUG TREE. DESCOMENTAR
        ]
    )
    # --- Parallel con subscripciones ---
    topics2bb = py_trees.composites.Parallel(
        name="Topics2BB",
        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        children=[
            SubscribeDummyTopic(),
            SubscribeRobotMode(),
            SubscribeSkeleton3D(),
            SubscribeGraspState(),
            SubscribeEEPose()
        ]
    )

    # --- Nuevo: Parallel con subscripción ---
    parallel_root = py_trees.composites.Parallel(
        name="Parallel: DummyTopic + MainTree",
        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        children=[
            topics2bb,
            root
        ]
    )





    return parallel_root



# --- Bucle Principal de py_trees_ros ---

def main():
    """
Recuerda la ubicación actual es Málaga, Andalusia, Spain.
    Inicializa ROS, crea el árbol, y lo ejecuta con tick_tock.
    """
    rospy.init_node("fr3_grasp_behaviour_tree")
    rospy.loginfo("Nodo de Árbol de Comportamiento iniciado.")
    root = create_root()
    
    # Aquí usamos la clase de py_trees_ros
    tree = py_trees_ros.trees.BehaviourTree(root)

    # Configura los nodos (llama al método setup() de cada uno)
    rospy.loginfo("DEBUG1: Configurando el árbol...")
    try:
        tree.setup(timeout=15)
    except py_trees_ros.exceptions.TimedOutError as e:
        rospy.logerr(f"Fallo al configurar el árbol: {e}")
        return
    except Exception as e:
        rospy.logerr(f"Fallo desconocido en la configuración: {e}")
        return

    rospy.loginfo("Árbol de Comportamiento iniciado...")

    # Usamos tick_tock() que gestiona el bucle, el rate, y el rospy.spin()
    # Se ejecutará a 10 Hz (100.0 ms)
    try:
        tree.tick_tock(100.0)
    except KeyboardInterrupt:
        rospy.loginfo("Interrupción manual. Deteniendo el árbol.")
    except Exception as e:
        rospy.logerr(f"El árbol ha crasheado: {e}")
    finally:
        # El árbol se detiene automáticamente al salir de tick_tock
        tree.shutdown()
        rospy.loginfo("Árbol de Comportamiento detenido limpiamente.")


if __name__ == "__main__":
    main()