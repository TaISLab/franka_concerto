#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación del BT de agarre de muñeca con py_trees_ros.
Estructura: Try-Catch-Finally robusta.
Stack: ROS 1 Noetic, Python, py_trees, py_trees_ros
"""

# Basics
import rospy
import numpy as np
import py_trees
import py_trees_ros
import time
import math
from scipy.spatial.transform import Rotation as R
import actionlib
from collections import deque

# Mensajes
import std_msgs.msg
import franka_msgs.msg
from skeleton_3d.msg import Skeleton3D
from geometry_msgs.msg import PoseStamped
from franka_buttons.msg import FrankaButtons

from actionlib_msgs.msg import GoalStatus
from franka_concerto.msg import MoveFR3Action, MoveFR3Goal # Acción de movimiento del FR3

# Servicios
from gripper_4f.srv import SetPWM, SetPWMResponse
from std_srvs.srv import Trigger, TriggerResponse

# Funciones y clases auxiliares
from franka_concerto.visualization_utils import VectorVisualizer, PointVisualizer
from franka_concerto.funciones_utiles import calculate_gripper_position_forearm_correction, calculate_quaternion_0_F

ROSBAG_PLAYING = False # Evitar usar timestamps del bag para filtrar datos antiguos

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
            # return py_trees.common.Status.SUCCESS
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
            # return py_trees.common.Status.SUCCESS
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
        self.kp_timestamp = None

    def setup(self, timeout):
        self.subscriber = rospy.Subscriber(self.topic_name, Skeleton3D, self.callback)
        try:
            # Visualizadores para RViz (vector normal y punto de muñeca)
            self.vec_vis = VectorVisualizer(topic_name="/wrist_normal_marker", frame_id="fr3_link0")
            self.point_vis = PointVisualizer(topic_name="/wrist_point_marker", frame_id="fr3_link0")
        except Exception as e:
            rospy.logwarn(f"[SubscribeSkeleton3D] No se pudo crear visualizadores RViz: {e}")
            self.vec_vis = None
            self.point_vis = None
        return True

    def callback(self, msg):
        # Extrae los keypoints relevantes
        if ROSBAG_PLAYING:
            self.kp_timestamp = rospy.Time.now()
        else: 
            self.kp_timestamp = msg.header.stamp # para filtrar datos antiguos

        kp_rwrist = msg.keypoints[10]
        kp_relbow = msg.keypoints[8]
        kp_rshoulder = msg.keypoints[6]

        # Función de validación: devuelve False si todos los componentes son (casi) cero
        def _kp_valid(kp, eps=1e-6):
            return not (abs(kp.x) < eps and abs(kp.y) < eps and abs(kp.z) < eps)

        # Si alguno de los keypoints es inválido, ignoramos esta lectura y no actualizamos el blackboard
        if not (_kp_valid(kp_rwrist) and _kp_valid(kp_relbow) and _kp_valid(kp_rshoulder)):
            rospy.logwarn_throttle(10, "[SubscribeSkeleton3D] Ignorando keypoints inválidos (todos ceros)")
            # Evitar usar datos antiguos: marcar como None
            self.rwrist = None
            self.relbow = None
            self.rshoulder = None
            if hasattr(self, "normal"):
                del self.normal
            return

        # Keypoints válidos: guardarlos y calcular la normal
        self.rwrist = kp_rwrist
        self.relbow = kp_relbow
        self.rshoulder = kp_rshoulder

        p1 = np.array([self.rshoulder.x, self.rshoulder.y, self.rshoulder.z])
        p2 = np.array([self.relbow.x, self.relbow.y, self.relbow.z])
        p3 = np.array([self.rwrist.x, self.rwrist.y, self.rwrist.z])
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            normal = normal / norm
        else:
            rospy.logwarn_throttle(10, "[SubscribeSkeleton3D] Vector normal casi cero; usando fallback (0,0,1)")
            normal = np.array([0.0, 0.0, 1.0])
        self.normal = normal
        # Publicar visualizaciones en RViz si están disponibles
        try:
            if hasattr(self, 'vec_vis') and self.vec_vis is not None:
                origin = [self.rwrist.x, self.rwrist.y, self.rwrist.z]
                # Color verde para el vector normal
                self.vec_vis.publish_vector(origin, normal, color=(0.0, 1.0, 0.0), scale=0.2)
            if hasattr(self, 'point_vis') and self.point_vis is not None:
                self.point_vis.publish_point(np.array([self.rwrist.x, self.rwrist.y, self.rwrist.z]), color=(1.0, 0.0, 0.0), scale=0.03)
        except Exception as e:
            rospy.logwarn_throttle(30, f"[SubscribeSkeleton3D] Error publicando en RViz: {e}")

        

    def update(self):
        if self.rshoulder is not None and self.relbow is not None and self.rwrist is not None and hasattr(self, "normal"):
            py_trees.blackboard.Blackboard().set("rshoulder_kp", self.rshoulder)
            py_trees.blackboard.Blackboard().set("relbow_kp", self.relbow)
            py_trees.blackboard.Blackboard().set("rwrist_kp", self.rwrist)
            py_trees.blackboard.Blackboard().set("wrist_normal", self.normal)
            py_trees.blackboard.Blackboard().set("skeleton_timestamp", self.kp_timestamp)
            # return py_trees.common.Status.SUCCESS
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
            # return py_trees.common.Status.SUCCESS
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
            # return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class SubscribeFrankaButtons(py_trees.behaviour.Behaviour):
    """
    Subscribe to /franka_buttons (franka_buttons/FrankaButtons)
    """
    def __init__(self, name="SubscribeFrankaButtons"):
        super(SubscribeFrankaButtons, self).__init__(name="SubscribeFrankaButtons")
        self.blackboard = py_trees.blackboard.Blackboard()
        
        self.topic_name = "/franka_buttons"
        self.subscriber = None
        self.cross = None

    def setup(self, timeout):
        self.subscriber = rospy.Subscriber(self.topic_name, FrankaButtons, self.callback)
        return True
    
    def callback(self, msg):
        self.cross = msg.cross

        if self.cross == True:
            rospy.logwarn(" Button CROSS pressed.")
            self.blackboard.set("homing_done", False)
            self.blackboard.set("approach_done", False)
            self.blackboard.set("contact_done", False)
            self.blackboard.set("cancel_task", True)

    
    def update(self):
        
        return py_trees.common.Status.RUNNING

# --- Neural Network Node ---
class CallNNService(py_trees.behaviour.Behaviour):

    def __init__(self, name="CallNNservice"):
        super(CallNNService, self).__init__(name)
        self.infer_value = None
    
    def setup(self, timeout):
        rospy.wait_for_service('/pronation_inference_node/infer_pronation')
        self.infer_service = rospy.ServiceProxy('/pronation_inference_node/infer_pronation', Trigger)
        return True
    
    def initialise(self):
        try:
            response = self.infer_service()
            if response.success:
                self.infer_value = float(response.message)  # Asumimos que el mensaje contiene el valor inferido
                rospy.loginfo(f"[{self.name}] NN Inference Success: {self.infer_value}")
            else:
                self.infer_value = None
                rospy.logerr(f"[{self.name}] NN Inference Failed!")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            self.infer_value = None

    def update(self):
        if self.infer_value is not None:
            py_trees.blackboard.Blackboard().set("nn_infer_value", self.infer_value)
            rospy.loginfo(f"[{self.name}] NN Inference Value set on blackboard: {self.infer_value}")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
    


# --- Hojas de Pre-checks ---

class RobotModeOk(py_trees.behaviour.Behaviour):
    """(Condición) Robot_mode ok?"""
    def __init__(self, name="RobotModeOk"):
        super(RobotModeOk, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.mode_required_list = [1, 2, 3, 4]  # Modo de ejecución [luz verde = 1]
    
    def update(self):
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
        # TODO: Comprobar el estado de la garra
        # Simulación: Falla la primera vez para forzar la acción
        
        if self.blackboard.get("gripper_closed") == True: # garra cerrada
            rospy.loginfo("Check: Gripper Open? -> No")
            return py_trees.common.Status.FAILURE
        
        rospy.loginfo("Check: Gripper Open? -> YES")
        return py_trees.common.Status.SUCCESS

class GripperPWMAction(py_trees.behaviour.Behaviour):
    """Acción genérica para controlar la garra por PWM."""
    def __init__(self, name="GripperPWMAction", pwm_value=-100, closed_state=False):
        super(GripperPWMAction, self).__init__(name)
        self.pwm_value = pwm_value
        self.closed_state = closed_state
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, timeout):
        rospy.wait_for_service('/gripper_4f/set_pwm')
        self.set_pwm_service = rospy.ServiceProxy('/gripper_4f/set_pwm', SetPWM)
        return super().setup(timeout)

    def initialise(self):
        try:
            response = self.set_pwm_service(self.pwm_value)
            if response.success:
                self.service_call_succeeded = True
                rospy.loginfo(f"[{self.name}] Gripper PWM {self.pwm_value} Command Sent!")
            else:
                self.service_call_succeeded = False
                rospy.logerr(f"[{self.name}] Gripper PWM {self.pwm_value} Command Failed!")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def update(self):
        if not self.service_call_succeeded:
            return py_trees.common.Status.FAILURE

        gripper_closed_status = self.blackboard.get("gripper_closed")
        if gripper_closed_status == self.closed_state:
            rospy.loginfo(f"[{self.name}] Gripper state reached ({self.closed_state})!")
            return py_trees.common.Status.SUCCESS
        elif gripper_closed_status is not None:
            return py_trees.common.Status.RUNNING
        else:
            rospy.logdebug(f"[{self.name}] Esperando datos en blackboard 'gripper_closed'...")
            return py_trees.common.Status.RUNNING

# Especializaciones
class OpenGripper(GripperPWMAction):
    def __init__(self, name="OpenGripper"):
        super(OpenGripper, self).__init__(name=name, pwm_value=-100, closed_state=False)

class CloseGripper(GripperPWMAction):
    def __init__(self, name="CloseGripper"):
        super(CloseGripper, self).__init__(name=name, pwm_value=300, closed_state=True)

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
        bb = py_trees.blackboard.Blackboard()
        if bb.get("homing_done") is True:
            rospy.loginfo(f"[{self.name}] Initial homing already done -> SUCCESS")
            return py_trees.common.Status.SUCCESS

        ee_pose = bb.get("ee_pose")
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

        # rospy.loginfo(f"AtHome check: pos_err={pos_err:.4f} m, ori_err={math.degrees(ori_err):.2f} deg (tol pos={self.pos_tol} m, tol ori={math.degrees(self.ori_tol_rad):.2f} deg)")

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
        bb = py_trees.blackboard.Blackboard()
        try:
            if state == GoalStatus.SUCCEEDED:
                self._succeeded = True
                bb.set("homing_done", True)
                rospy.loginfo(f"[{self.name}] Acción completada: SUCCEEDED")
            else:
                self._succeeded = False
                bb.set("homing_done", False)
                rospy.logwarn(f"[{self.name}] Acción terminada con estado {state}")
        except Exception:
            self._succeeded = False
            bb.set("homing_done", False)
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
    # TODO: Filtrar datos antiguos y comprobar que es estable
    def __init__(self, name="IsWristDataAvailable"):
        super(IsWristDataAvailable, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        # DEBUG_memory: No necesario si usamos memory=True en el Sequence padre
        if self.blackboard.get("approach_done") is True:
            return py_trees.common.Status.SUCCESS
        
        rospy.loginfo("Check: Wrist Data Available?")
        rwrist = self.blackboard.get("rwrist_kp")
        normal = self.blackboard.get("wrist_normal")
        kp_timestamp = self.blackboard.get("skeleton_timestamp")

        time_now = rospy.Time.now()
        time_delay = (time_now - kp_timestamp).to_sec() if kp_timestamp is not None else None

        rospy.loginfo(f"Time delay since last wrist data: {time_delay} seconds")
        # Filtrar datos antiguos (más de 1 segundo)
        if kp_timestamp is None or (time_now - kp_timestamp).to_sec() > 1.0:
            return py_trees.common.Status.FAILURE
        
        if rwrist is None or normal is None:
            return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.SUCCESS

class IsWristStable(py_trees.behaviour.Behaviour):
    """(Condición) Muñeca estable durante un intervalo de tiempo.

    Comprueba estabilidad usando las últimas lecturas de `rwrist_kp` (posición)
    y `wrist_normal` (vector normal). Usa una ventana temporal `window_sec`.
    """
    def __init__(self, name="IsWristStable", window_sec=2.0, pos_tol=0.02, ori_tol_deg=5.0, min_samples=5):
        super(IsWristStable, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.window_sec = float(window_sec)
        self.pos_tol = float(pos_tol)
        self.ori_tol_deg = float(ori_tol_deg)
        self.min_samples = int(min_samples)
        # buffer: deque of tuples (rospy.Time, np_pos(3,), np_normal(3,))
        self.buffer = deque()

    def initialise(self):
        # No vaciamos el buffer: queremos conservar historial entre inicializaciones opcionales,
        # pero en caso de querer resetarlo por cada entry, descomenta la siguiente línea.
        # self.buffer.clear()

        rospy.logdebug(f"[{self.name}] Initialise (window={self.window_sec}s, pos_tol={self.pos_tol}m, ori_tol={self.ori_tol_deg}deg)")

    def _prune_buffer(self, now):
        """Eliminar entradas más antiguas que now - window_sec."""
        cutoff = now - rospy.Duration(self.window_sec)
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def _vec_from_kp(self, kp):
        try:
            return np.array([kp.x, kp.y, kp.z], dtype=float)
        except Exception:
            return None

    def _normal_from_bb(self, normal_bb):
        try:
            arr = np.array(normal_bb, dtype=float)
            norm = np.linalg.norm(arr)
            if norm <= 1e-8:
                return None
            return arr / norm
        except Exception:
            return None

    def update(self):
        # DEBUG_memory: No necesario si usamos memory=True en el Sequence padre
        # if self.blackboard.get("approach_done") is True:
        #     return py_trees.common.Status.SUCCESS
        
        # Leer del blackboard
        rwrist = self.blackboard.get("rwrist_kp")
        wrist_normal = self.blackboard.get("wrist_normal")
        kp_timestamp = self.blackboard.get("skeleton_timestamp")  # asumimos que lo publicas

        if rwrist is None or wrist_normal is None or kp_timestamp is None:
            rospy.logdebug(f"[{self.name}] Datos insuficientes en blackboard (rwrist/wrist_normal/timestamp).")
            return py_trees.common.Status.RUNNING

        # convertir
        pos = self._vec_from_kp(rwrist)
        normal = self._normal_from_bb(wrist_normal)

        if pos is None or normal is None:
            rospy.logwarn_throttle(5, f"[{self.name}] Lectura inválida (pos o normal nula).")
            return py_trees.common.Status.RUNNING

        now = rospy.Time.now()
        # Guardar entrada (usamos kp_timestamp si quieres usar la marca de la cámara; aquí guardamos ahora)
        self.buffer.append((now, pos, normal))
        self._prune_buffer(now)

        if len(self.buffer) < self.min_samples:
            rospy.logdebug(f"[{self.name}] Muestras insuficientes ({len(self.buffer)}/{self.min_samples}).")
            return py_trees.common.Status.RUNNING

        # Calcular medida de posición: usar posición media o referencia inicial
        positions = np.vstack([entry[1] for entry in self.buffer])
        mean_pos = np.mean(positions, axis=0)
        max_disp = np.max(np.linalg.norm(positions - mean_pos, axis=1))

        # Calcular medida de orientación: máximo ángulo entre normales y la media
        normals = np.vstack([entry[2] for entry in self.buffer])
        # compute pairwise max angle relative to mean normal
        mean_normal = np.mean(normals, axis=0)
        mn_norm = np.linalg.norm(mean_normal)
        if mn_norm <= 1e-8:
            rospy.logwarn_throttle(5, f"[{self.name}] Mean normal casi cero, considerándolo inestable.")
            return py_trees.common.Status.RUNNING
        mean_normal = mean_normal / mn_norm
        # angles in degrees between each normal and mean_normal
        dots = np.clip(np.dot(normals, mean_normal), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        max_angle = float(np.max(angles))

        # rospy.logdebug(f"[{self.name}] max_disp={max_disp:.4f} m, max_angle={max_angle:.2f} deg, samples={len(self.buffer)}")

        if max_disp <= self.pos_tol and max_angle <= self.ori_tol_deg:
            rospy.loginfo(f"[{self.name}] Wrist STABLE")
            return py_trees.common.Status.SUCCESS
        else:
            rospy.logdebug(f"[{self.name}] Wrist not stable yet")
            return py_trees.common.Status.RUNNING

class PlanAndApproach(py_trees.behaviour.Behaviour):
    """(Acción) Plan y aprox — calcula approach = kp_wrist + normal*approach_distance y envía ese pose como goal (IMPEDANCE_MOVE)."""
    def __init__(self, name="PlanAndApproach", approach_distance=0.15, forearm_correction=-0.1, max_velocity=0.1, task_type=0):
        super(PlanAndApproach, self).__init__(name)
        self.done = False
        self._client = None
        self._goal_sent = False
        self._succeeded = False
        self._server_available = False
        self.blackboard = py_trees.blackboard.Blackboard()
        self.approach_distance = approach_distance
        self.forearm_correction = forearm_correction
        self.max_velocity = max_velocity
        self.task_type = 0

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

        # DEBUG_memory: No necesario si usamos memory=True en el Sequence padre
        # if self.blackboard.get("approach_done") is True:
        #     return py_trees.common.Status.SUCCESS

        if self._goal_sent and not self.done:
            rospy.logdebug(f"[{self.name}] initialise called but goal already sent and not done; skipping re-send")
            return

        self.done = False
        self._succeeded = False

        if not self._server_available:
            rospy.logerr(f"[{self.name}] Servidor de acción no disponible.")
            return

        # Si el cliente ya tiene un goal PENDING/ACTIVE, no reenviamos
        try:
            state = self._client.get_state()
            if state in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                rospy.loginfo(f"[{self.name}] Action client already has active goal (state={state}), not resending.")
                self._goal_sent = True
                return
        except Exception:
            # si no podemos consultar, intentamos de todos modos (log)
            rospy.logdebug(f"[{self.name}] No se pudo consultar estado del action client antes de enviar.")

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

        # calcular pose de aproximación (kp_wrist + normal * approach_distance) con corrección de antebrazo
        try:
            approach_point = calculate_gripper_position_forearm_correction(
                rw, normal_arr, forearm, self.approach_distance, self.forearm_correction
            )
            self.blackboard.set("approach_wrist_position", rw)
            self.blackboard.set("approach_normal_vector", normal_arr)
            self.blackboard.set("approach_forearm_vector", forearm)


        except Exception as e:
            rospy.logerr(f"[{self.name}] Error calculando posición de approach: {e}")
            return

        # calcular orientación (igual que hri_states_machine)
        try:
            quat_pose = calculate_quaternion_0_F(-forearm, -normal_arr)
            qx = quat_pose.x
            qy = quat_pose.y
            qz = quat_pose.z
            qw = quat_pose.w
        except Exception as e:
            rospy.logerr(f"[{self.name}] Error calculando orientación: {e}")
            return

        rospy.loginfo(f"[{self.name}] Approach pose: x={approach_point.x:.3f}, y={approach_point.y:.3f}, z={approach_point.z:.3f}")

        # Construir y enviar goal con la pose calculada (IMPEDANCE_MOVE)
        goal = MoveFR3Goal()
        goal.task_type = self.task_type # TASK_IMPEDANCE_MOVE (usar 1 según action spec)
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "fr3_link0"
        goal.target_pose.pose.position.x = approach_point.x
        goal.target_pose.pose.position.y = approach_point.y
        goal.target_pose.pose.position.z = approach_point.z
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        goal.max_velocity = self.max_velocity

        try:
            self._client.send_goal(goal, done_cb=self._done_cb, feedback_cb=self._feedback_cb)
            self._goal_sent = True
            rospy.loginfo(f"[{self.name}] Goal (approach pose) enviado a fr3_motion_server.")
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
                self.blackboard.set("approach_done", True)
                rospy.logwarn(f"[{self.name}] RRR Acción completada: SUCCEEDED")

            else:
                self._succeeded = False
                rospy.logwarn(f"[{self.name}] Acción terminada con estado {state}")
        except Exception:
            self._succeeded = False
        finally:
            # marcar como finalizado y permitir reenvío futuro
            self.done = True
            self._goal_sent = False

    def update(self):

        # Verificar self.done y self._succeeded
        if self.done: # Si terminó, devolver SUCCESS/FAILURE según el resultado
            if self._succeeded:
                rospy.loginfo(f"[{self.name}] done -> returning SUCCESS")
                self.blackboard.set("approach_done", True)
                return py_trees.common.Status.SUCCESS
            else:
                rospy.loginfo(f"[{self.name}] done -> returning FAILURE")
                self.blackboard.set("approach_done", False)
                return py_trees.common.Status.FAILURE
            
        rospy.loginfo(f"[{self.name}] RUNNING")
        return py_trees.common.Status.RUNNING
        
    def terminate(self, new_status):
        # Si se interrumpe la acción, cancelar goal activo
        if self._goal_sent and not self.done:
            try:
                self._client.cancel_goal()
                rospy.loginfo(f"[{self.name}] Goal cancelado debido a terminación con nuevo estado {new_status}.")
            except Exception as e:
                rospy.logerr(f"[{self.name}] Error cancelando goal en terminate: {e}")


class PlanAndContact(py_trees.behaviour.Behaviour):
    """Plan y contacto:  pero con approach_distance=0.0 y permite especificar wrist/normal/forearm."""
    def __init__(self, name="PlanAndContact"):
        super(PlanAndContact, self).__init__(name=name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.approach_wrist = None
        self.approach_normal = None
        self.approach_forearm = None
        self.task_type = 0
        self.approach_distance = 0.0
        self.forearm_correction = -0.1
        self.max_velocity = 0.1
        self.done = False

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
        # IMPORTANT: do not force-reset _goal_sent here; that was causing re-sends
        # sólo inicializamos la ejecución si no hay un goal activo
        if self.blackboard.get("contact_done") is True:
            return py_trees.common.Status.SUCCESS
        
        if self.approach_wrist is None or self.approach_normal is None or self.approach_forearm is None:
            self.approach_wrist = self.blackboard.get("approach_wrist_position")
            self.approach_normal = self.blackboard.get("approach_normal_vector")
            self.approach_forearm = self.blackboard.get("approach_forearm_vector")

        self.done = False
        self._succeeded = False

        if not self._server_available:
            rospy.logerr(f"[{self.name}] Servidor de acción no disponible.")
            return

        # Si el cliente ya tiene un goal PENDING/ACTIVE, no reenviamos
        try:
            state = self._client.get_state()
            if state in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                rospy.loginfo(f"[{self.name}] Action client already has active goal (state={state}), not resending.")
                self._goal_sent = True
                return
        except Exception:
            # si no podemos consultar, intentamos de todos modos (log)
            rospy.logdebug(f"[{self.name}] No se pudo consultar estado del action client antes de enviar.")

        # calcular pose de aproximación (kp_wrist + normal * approach_distance) con corrección de antebrazo
        try:
            contact_point = calculate_gripper_position_forearm_correction(
                self.approach_wrist, self.approach_normal, self.approach_forearm, 0.0, -0.10
            )

        except Exception as e:
            rospy.logerr(f"[{self.name}] Error calculando posición de contacto: {e}")
            return

        # calcular orientación (igual que hri_states_machine)
        try:
            quat_pose = calculate_quaternion_0_F(-self.approach_forearm, -self.approach_normal)
            qx = quat_pose.x
            qy = quat_pose.y
            qz = quat_pose.z
            qw = quat_pose.w
        except Exception as e:
            rospy.logerr(f"[{self.name}] Error calculando orientación: {e}")
            return

        rospy.loginfo(f"[{self.name}] Approach pose: x={contact_point.x:.3f}, y={contact_point.y:.3f}, z={contact_point.z:.3f}")

        # Construir y enviar goal con la pose calculada (IMPEDANCE_MOVE)
        goal = MoveFR3Goal()
        goal.task_type = self.task_type  # TASK_IMPEDANCE_MOVE (usar 1 según action spec)
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "fr3_link0"
        goal.target_pose.pose.position.x = contact_point.x
        goal.target_pose.pose.position.y = contact_point.y
        goal.target_pose.pose.position.z = contact_point.z
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        goal.max_velocity = self.max_velocity

        try:
            self._client.send_goal(goal, done_cb=self._done_cb, feedback_cb=self._feedback_cb)
            self._goal_sent = True
            rospy.loginfo(f"[{self.name}] Goal (approach pose) enviado a fr3_motion_server.")
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
                self.blackboard.set("contact_done", True)
                rospy.loginfo(f"[{self.name}] Acción completada: SUCCEEDED")
            else:
                self._succeeded = False
                rospy.logwarn(f"[{self.name}] Acción terminada con estado {state}")
        except Exception:
            self._succeeded = False
        finally:
            # marcar como finalizado y permitir reenvío futuro
            self.done = True
            self._goal_sent = False

    def update(self):
        if not self._server_available:
            rospy.logwarn(f"[{self.name}] server not available in update()")
            return py_trees.common.Status.FAILURE

        # Si hay un goal en el cliente y sigue activo/pending, devolver RUNNING
        try:
            if self._client is not None:
                state = self._client.get_state()
                if state in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                    rospy.logdebug(f"[{self.name}] Action state {state} -> RUNNING")
                    return py_trees.common.Status.RUNNING
                # Mapear estados terminales si el callback no se ha disparado aún
                if state == GoalStatus.SUCCEEDED:
                    self._succeeded = True
                    self.done = True
                elif state in (GoalStatus.ABORTED, GoalStatus.PREEMPTED, GoalStatus.REJECTED,
                               GoalStatus.RECALLED, GoalStatus.LOST):
                    self._succeeded = False
                    self.done = True
        except Exception as e:
            rospy.logwarn(f"[{self.name}] No se pudo consultar estado del action client: {e}")

        # Si enviamos un goal y aún no terminó, seguir RUNNING
        if self._goal_sent and not self.done:
            rospy.logdebug(f"[{self.name}] goal_sent and not done -> RUNNING")
            return py_trees.common.Status.RUNNING

        # Si terminó, devolver SUCCESS/FAILURE según el resultado
        if self.done:
            rospy.logdebug(f"[{self.name}] done -> returning final status ({self._succeeded})")
            # limpiar flags para permitir reintentos posteriores
            final = py_trees.common.Status.SUCCESS if self._succeeded else py_trees.common.Status.FAILURE
            self._goal_sent = False
            return final

        # Caso por defecto: RUNNING (evita reentradas que provoquen initialise/reenvío)
        rospy.logdebug(f"[{self.name}] default RUNNING")
        return py_trees.common.Status.RUNNING

class Timer(py_trees.behaviour.Behaviour):
    """(Acción) Temporizador de timeout segundos."""
    def __init__(self, name="Timer", timeout=5.0):
        super(Timer, self).__init__(name)
        self.timeout = float(timeout)
        self.start_time = None

    def initialise(self):
        self.start_time = rospy.Time.now()
        rospy.loginfo(f"[{self.name}] Waiting timeout={self.timeout}s...")

    def update(self):
        # Comprobar timeout
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed >= self.timeout:
            rospy.logwarn(f"[{self.name}] Timeout waiting for gripper to close.")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class CancelIfPressCrossButton(py_trees.behaviour.Behaviour):
    """(Acción) Comprueba el valor de cancel_task en el blackboard y devuelve SUCCESS si es False y Failure si es True."""
    def __init__(self, name="CancelIfPressCrossButton"):
        super(CancelIfPressCrossButton, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
    
    def update(self):
        cancel_task = self.blackboard.get("cancel_task")
        if cancel_task is True:
            rospy.logwarn("Check: Cross Button Pressed! Cancelling task...")
            return py_trees.common.Status.FAILURE
        else:
            # rospy.logwarn("Check: Cross Button Not Pressed. Continuing task...")
            return py_trees.common.Status.SUCCESS

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

class ResetCancelTaskButton(py_trees.behaviour.Behaviour):
    """(Acción) Resetea el botón de parada de emergencia."""
    def __init__(self, name="ResetCancelTaskButton"):
        super(ResetCancelTaskButton, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def initialise(self):
        rospy.loginfo("Action: Resetting Cancel Task Button...")
        self.blackboard.set("cancel_task", False)
    
    def update(self):
        return py_trees.common.Status.SUCCESS

# --- Funciones de Construcción ---

def create_pre_checks_subtree():
    """Crea el sub-árbol de Pre-checks."""
    
    # ? (Selector) "Asegurar Garra Abierta"
    ensure_gripper_open = py_trees.composites.Selector(
        name="Asegurar Garra Abierta",
        memory=True,
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
            ensure_home_pose
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
            IsWristDataAvailable(),
            IsWristStable(window_sec=2.0, pos_tol=0.01, ori_tol_deg=3.0, min_samples=10),
            PlanAndApproach()
        ]
    )
    
    # -> (Sequence) "Contacto"
    contact_seq = py_trees.composites.Sequence(
        name="Contacto",
        memory=True,
        children=[
            PlanAndContact()
        ]
    )

    # -> (Sequence) "Agarre"
    grasp_seq = py_trees.composites.Sequence(
        name="Agarre",
        memory=True,
        children=[
            CloseGripper(),
            Timer(timeout=2.0),
            CallNNService(),
            Timer(timeout=2.0),
        ]
    )

    # -> (Sequence) "Tarea principal"
    main_task_root = py_trees.composites.Sequence(
        name="Tarea principal",
        memory=True,
        children=[
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
        memory=False,
        children=[
            # ensure_gripper_open,
            # ensure_retract,
            # ensure_home
            ResetCancelTaskButton()
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
        memory=False, # Memoria para la secuencia de tareas
        children=[
            CancelIfPressCrossButton(),
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
        memory=False,
        children=[
            try_catch_block,
            cleanup_subtree # DEBUG TREE. DESCOMENTAR
        ]
    )
    # --- Parallel con subscripciones ---
    topics2bb = py_trees.composites.Parallel(
        name="Topics2BB",
        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, # todos deben tener éxito
        children=[
            SubscribeDummyTopic(),
            SubscribeRobotMode(),
            SubscribeSkeleton3D(),
            SubscribeGraspState(),
            SubscribeEEPose(),
            SubscribeFrankaButtons()
        ]
    )

    # --- Nuevo: Parallel con subscripción ---
    parallel_root = py_trees.composites.Parallel(
        name="Parallel: DummyTopic + MainTree",
        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, # todos deben tener éxito
        children=[
            topics2bb,
            root
        ]
    )

    return parallel_root


# --- Bucle Principal de py_trees_ros ---

def main():
    """
    Inicializa ROS, crea el árbol, y lo ejecuta con tick_tock.
    """
    rospy.init_node("fr3_grasp_behaviour_tree", log_level=rospy.INFO)
    rospy.loginfo("Nodo de Árbol de Comportamiento iniciado.")
    
    py_bb = py_trees.blackboard.Blackboard()
    py_bb.set("homing_done", False) # Variable global para homing
    py_bb.set("approach_done", False)

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