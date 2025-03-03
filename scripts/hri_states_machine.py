#!/usr/bin/env python3
 
# Import the necessary libraries
import rospy
from smach import State, StateMachine
import smach_ros # Libreria Smach para ROS. Máquina de estados finitos
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from franka_buttons.msg import FrankaButtons
from std_msgs.msg import Int32
from skeleton_3d.msg import Skeleton3D
from funciones_utiles import are_kp_valid, calculate_normal_vector, print_pose_named, calculate_gripper_position, calculate_quaternion_0_F
from visualization_utils import VectorVisualizer

class VisionMonitor:
  """ Clase singleton para almacenar y compartir los KP globalmente """
  _instance = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super(VisionMonitor, cls).__new__(cls)
      cls._instance.init_ros()
    return cls._instance

  def init_ros(self):
    self.kp_R_Wrist = None
    self.kp_R_Elbow = None
    self.kp_R_Shoulder = None
    self.normal_vector = None
    self.vect_forearm = None
    self.kp_valid = False

    self.sub = rospy.Subscriber('/skeleton_3D', Skeleton3D, self.obtain_skeleton3D_callback)
    
    # Para DEBUG mediante Rviz. 
    self.normal_vector_visualizer = VectorVisualizer("/normal_vector_marker")
    self.forearm_vector_visualizer = VectorVisualizer("/forearm_vector_marker")
  

  def obtain_skeleton3D_callback(self, msg):
    """ Callback que recibe los datos de visión y los almacena globalmente """
    keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])

    self.kp_R_Shoulder = keypointsX[6]
    self.kp_R_Elbow = keypointsX[8]
    self.kp_R_Wrist = keypointsX[10]

    if are_kp_valid(self.kp_R_Shoulder, self.kp_R_Elbow, self.kp_R_Wrist):
      self.normal_vector = calculate_normal_vector(self.kp_R_Shoulder, self.kp_R_Elbow, self.kp_R_Wrist)
      self.vect_forearm = self.kp_R_Wrist - self.kp_R_Elbow
      self.vect_forearm = self.vect_forearm / np.linalg.norm(self.vect_forearm) # normalizar

      # DEBUG con RVIZ
      self.normal_vector_visualizer.publish_vector(self.kp_R_Wrist, self.normal_vector, color=(0.0, 0.0, 1.0), scale=0.2)
      self.forearm_vector_visualizer.publish_vector(self.kp_R_Wrist, self.vect_forearm, color=(1.0, 0.0, 0.0), scale=0.2)

      self.kp_valid = True
    else:
      self.kp_valid = False

  def get_latest_data(self):
    """
    Devuelve los últimos valores actualizados. Devuelve un diccionario.
    Se accede con: vision_data['kp_R_Wrist']
    """
    return {
      'kp_R_Wrist': self.kp_R_Wrist,
      'kp_R_Elbow': self.kp_R_Elbow,
      'kp_R_Shoulder': self.kp_R_Shoulder,
      'normal_vector': self.normal_vector,
      'vect_forearm': self.vect_forearm,
      'kp_valid': self.kp_valid
    }

class ButtonMonitor:
  """ Clase singleton para almacenar y compartir los botones globalmente """
  _instance = None

  def __new__(cls):
    if cls._instance is None:
        cls._instance = super(ButtonMonitor, cls).__new__(cls)
        cls._instance.init_ros()
    return cls._instance

  def init_ros(self):
    """ Inicializa la subscripción al tópico de los botones """
    self.check = False
    self.circle = False
    self.cross = False
    self.x = 0.0
    self.y = 0.0

    self.sub = rospy.Subscriber("/franka_buttons", FrankaButtons, self.button_callback)

  def button_callback(self, msg):
    """ Callback que recibe los datos de los botones y los almacena globalmente """
    self.check = msg.check
    self.circle = msg.circle
    self.cross = msg.cross
    self.x = msg.x
    self.y = msg.y

  def get_latest_data(self):
    """ Devuelve los últimos valores actualizados de los botones """
    return {
        'check': self.check,
        'circle': self.circle,
        'cross': self.cross,
        'x': self.x,
        'y': self.y
    }

class GripperManager:
  """ Singleton para manejar el estado y control de la garra """
  _instance = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super(GripperManager, cls).__new__(cls)
      cls._instance.init_ros()
    return cls._instance

  def init_ros(self):
    """ Inicialización de ROS """
    self.grip_state = -1  # False = Cerrado, True = Abierto
    # self.gripper_state_d = -1  # False = Cerrado, True = Abierto
    self.sub = rospy.Subscriber("/grip_state", Int32, self.grip_callback)
    self.pub = rospy.Publisher('/gripper_state_desired', Int32, queue_size=10)

  def grip_callback(self, msg):
    """ Callback que actualiza el estado de la garra
    -1 - Inicializada pero no se ha obtenido el primer mensaje
     0 - Abierta
     1 - Cerrada con flexión
     2 - Cerrada modo pinza
    """
    self.grip_state = msg.data

  def open_gripper(self):
    """ Publica el comando para abrir la garra """
    rospy.loginfo("Enviando comando para ABRIR la garra")
    self.pub.publish(0)

  def close_gripper(self):
    """ Publica el comando para cerrar la garra """
    rospy.loginfo("Enviando comando para CERRAR la garra")
    self.pub.publish(1)

  def get_state(self):
    """ Devuelve el estado actual de la garra """
    return self.grip_state

# Definir estado REPOSO
class Reposo(State):
  def __init__(self, buttons):
    State.__init__(self, outcomes=['buscar','esperando'])

    self.buttons = buttons  # Guardamos la instancia
    
    # publica la pose inicial
    self.desired_pose_publisher = rospy.Publisher("/desired_pose", PoseStamped, queue_size=10)
    self.init_pose = PoseStamped()

  def execute(self, userdata):
    
    rospy.loginfo('Ejecutando estado de REPOSO...')

    # Mover manipulador a la pose de reposo
    self.init_pose.header.frame_id = "fr3_link0"
    self.init_pose.pose.position = Point(0.13, 0.0, 0.73)
    self.init_pose.pose.orientation = Quaternion(1.0, 0.0, 0.0, 0.0)
    
    rospy.logdebug("Publicando pose de reposo")
    self.desired_pose_publisher.publish(self.init_pose)

    # Obtener estado de los botones
    buttons_data = self.buttons.get_latest_data()

    if buttons_data['check']:
      return 'buscar'
    
    rospy.sleep(1)
    return 'esperando'  # O algún estado seguro

# Estado BUSCANDO. El sistema busca que la muñeca esté dentro del espacio de trabajo
class Buscando(State):
  def __init__(self, vision, buttons):
    State.__init__(self, outcomes=['encontrado','no_encontrado', 'abortar'])
    
    self.vision = vision  # Guardamos la instancia
    self.buttons = buttons  # Guardamos la instancia
    
    self.tiempo_inicio = None  # Guarda el tiempo en que el keypoint entra en la zona

  def esta_dentro_del_espacio(self, punto):
    """ Verifica si el punto está dentro del volumen cartesiano definido """
    x_min, x_max = 0.0, 0.55
    y_min, y_max = -0.3, 0.3
    z_min, z_max = 0.0, 0.8

    return (x_min <= punto[0] <= x_max and 
            y_min <= punto[1] <= y_max and 
            z_min <= punto[2] <= z_max)

  def execute(self, userdata):
    rospy.loginfo("Ejecutando estado BUSCANDO...")
    ################################
    # IMPORTANTE!! Modificar para que estudie que la muñeca permanece quieta en el espacio
    ###############################
    
    vision_data = self.vision.get_latest_data()
    buttons_data = self.buttons.get_latest_data()

    # if (vision_data['kp_valid']) and (self.esta_dentro_del_espacio(vision_data['kp_R_Wrist'])):
    #   rospy.logwarn("## Dentro del espacio de busqueda ##")

    #   if self.tiempo_inicio is None:
    #     self.tiempo_inicio = time.time()  # inicio del contador

    #   if time.time() - self.tiempo_inicio >= 2.0:
    #     rospy.loginfo("KP ha permanecido 2 segundos en el espacio de busqueda.")
    #     return 'encontrado'  # al siguiente estado  

    if buttons_data['circle']:
      return 'encontrado'


    if buttons_data['cross']:
      return 'abortar'

    rospy.loginfo("## No encontrado en el espacio de busqueda ##")
    rospy.sleep(1)  # pausa
    return 'no_encontrado'

class Aproximando(State):
  def __init__(self, vision, buttons):
    State.__init__(self, 
                   outcomes=['aprox_completada','abortar'],
                   output_keys=['last_kp_R_Wrist', 'last_normal_vector'])  # Exporta estos valores

    self.vision = vision  # instancia del monitor de vision
    self.buttons = buttons  # instancia del monitor de los botones

    # Subscribers
    self.current_pose_subscriber = rospy.Subscriber("/current_pose", PoseStamped, self.current_pose_callback) # subscripción al current_pose
    self.path_planning_state_subscriber = rospy.Subscriber("/path_planner_state", Int32, self.path_planning_state_callback) # subscripción al estado de la planificación

    # Publishers
    self.desired_pose_publisher = rospy.Publisher("/desired_pose", PoseStamped, queue_size=10) # publica en /desired_pose, entrada del planificador de trayectorias

    self.current_pose = PoseStamped()
    self.path_planning_state = -1

    self.vector_visualizer = VectorVisualizer()  # Crear instancia

  def current_pose_callback(self, msg):
    rospy.logdebug("Pose actual recibida")
    self.current_pose = msg

  def path_planning_state_callback(self, msg):
    self.path_planning_state = msg.data
    # rospy.logwarn(f"FSM: Estado de la trayectoria recibido: {msg.data}")
  
  def execute(self, userdata):

    rospy.loginfo("Ejecutando estado Aproximando")
    rospy.loginfo("Espera de 2 segundos")
    rospy.sleep(2) # Esta espera es para que el humano ponga la mano en el punto deseado de agarre
    
    distancia_aproximacion = 0.2
    vision_data = self.vision.get_latest_data()
    buttons_data = self.buttons.get_latest_data()

    userdata.last_kp_R_Wrist = vision_data['kp_R_Wrist']  # Guardar valor para el siguiente estado
    userdata.last_normal_vector = vision_data['normal_vector']

    self.vector_visualizer.publish_vector(vision_data['kp_R_Wrist'], vision_data['normal_vector'], color=(1.0, 0.0, 0.0), scale=0.2)

    desired_pose = PoseStamped()

    desired_pose.pose.position = calculate_gripper_position(vision_data['kp_R_Wrist'], vision_data['normal_vector'], distancia_aproximacion)
    desired_pose.pose.orientation = calculate_quaternion_0_F(-vision_data['vect_forearm'], -vision_data['normal_vector']) # invertir el sentido de los vectores de entrada
    self.desired_pose_publisher.publish(desired_pose) # publicar el kp q le pasa el estado anterior

    rospy.loginfo("Ejecutando trayectoria...")
    self.desired_pose_publisher.publish(desired_pose) # publicar el kp q le pasa el estado anterior

    init_time = rospy.Time.now().to_sec()  # Convertir a segundos

    while self.path_planning_state != 2:
      rospy.sleep(0.1)

      if rospy.Time.now().to_sec() - init_time > 10:  # Restar en segundos
        rospy.logerr("Error en la trayectoria")
        # return 'abortar'
        break # DEBUG porq no se verifica bien la finalización de trayectorias
      
    rospy.loginfo("Trayectoria completada")
    
    return 'aprox_completada'
  
class Agarre(State):
  def __init__(self, buttons, gripper):
    State.__init__(self, 
                   outcomes=['agarrado','abortar'],
                   input_keys=['last_kp_R_Wrist', 'last_normal_vector'],
                   output_keys=['last_kp_R_Wrist', 'last_normal_vector'])

    # self.vision = vision  # instancia del monitor de vision
    self.buttons = buttons  # instancia del monitor de los botones
    self.gripper = gripper

    # subscripción al current_pose
    self.current_pose_subscriber = rospy.Subscriber("/current_pose", PoseStamped, self.current_pose_callback)
    
    # subscripción al estado de la planificación
    self.path_planning_state_subscriber = rospy.Subscriber("/path_planner_state", Int32, self.path_planning_state_callback)

    # publica en /desired_pose, entrada del planificador de trayectorias
    self.desired_pose_publisher = rospy.Publisher("/desired_pose", PoseStamped, queue_size=10)

    self.current_pose = PoseStamped()
    self.path_planning_state = -1

  def current_pose_callback(self, msg):
    rospy.logdebug("Pose actual recibida")
    self.current_pose = msg

  def path_planning_state_callback(self, msg):
    self.path_planning_state = msg.data
    # rospy.logwarn(f"FSM: Estado de la trayectoria recibido: {msg.data}")
  
  def execute(self, userdata):

    rospy.loginfo("Ejecutando estado Aproximando")
    rospy.loginfo("Espera de 2 segundos")
    # rospy.sleep(2) # Esta espera es para que el humano ponga la mano en el punto deseado de agarre
    
    distancia_aproximacion = 0
    buttons_data = self.buttons.get_latest_data()

    # self.gripper.open_gripper() # Abrir la pinza
    
    desired_pose = PoseStamped()
    desired_pose.pose.position = calculate_gripper_position(userdata.last_kp_R_Wrist, userdata.last_normal_vector, distancia_aproximacion)
    desired_pose.pose.orientation = self.current_pose.pose.orientation # copiamos orientación, debe ser correcta
    self.desired_pose_publisher.publish(desired_pose) # publicar el kp q le pasa el estado anterior

    rospy.loginfo("Ejecutando trayectoria...")
    self.desired_pose_publisher.publish(desired_pose) # publicar el kp q le pasa el estado anterior

    init_time = rospy.Time.now().to_sec()  # Convertir a segundos

    while self.path_planning_state != 2:
      rospy.sleep(0.1)

      if rospy.Time.now().to_sec() - init_time > 10:  # Restar en segundos
        rospy.logerr("Error en la trayectoria")
        # return 'abortar'
        break # debug porq no se verifica bien la finalización de trayectorias

    rospy.logwarn("Cerrar pinza")
    self.gripper.close_gripper() # Cerrar la pinza
    rospy.sleep(2)
    self.gripper.open_gripper()
    rospy.sleep(2)

    return 'agarrado'

class Retirada(State):
  def __init__(self):
    State.__init__(self, 
                   outcomes=['retirada','abortar'],
                   input_keys=['last_kp_R_Wrist', 'last_normal_vector'])  # Exporta estos valores

    # Subscribers
    self.current_pose_subscriber = rospy.Subscriber("/current_pose", PoseStamped, self.current_pose_callback) # subscripción al current_pose
    self.path_planning_state_subscriber = rospy.Subscriber("/path_planner_state", Int32, self.path_planning_state_callback) # subscripción al estado de la planificación
    
    # Publishers
    self.desired_pose_publisher = rospy.Publisher("/desired_pose", PoseStamped, queue_size=10) # publica en /desired_pose, entrada del planificador de trayectorias

    self.current_pose = PoseStamped()
    self.path_planning_state = -1

  def current_pose_callback(self, msg):
    rospy.logdebug("Pose actual recibida")
    self.current_pose = msg

  def path_planning_state_callback(self, msg):
    self.path_planning_state = msg.data
    # rospy.logwarn(f"FSM: Estado de la trayectoria recibido: {msg.data}")
  
  def execute(self, userdata):

    rospy.loginfo("Ejecutando estado Retirada")
    rospy.loginfo("Espera de 2 segundos")
    rospy.sleep(2) # Esta espera es para que el humano ponga la mano en el punto deseado de agarre
    
    distancia_aproximacion = 0.2

    desired_pose = PoseStamped()

    desired_pose.pose.position = calculate_gripper_position(userdata.last_kp_R_Wrist, userdata.last_normal_vector, distancia_aproximacion)
    desired_pose.pose.orientation = self.current_pose.pose.orientation # copiamos orientación, debe ser correcta
    self.desired_pose_publisher.publish(desired_pose) # publicar el kp q le pasa el estado anterior

    rospy.loginfo("Ejecutando trayectoria...")
    self.desired_pose_publisher.publish(desired_pose) # publicar el kp q le pasa el estado anterior

    init_time = rospy.Time.now().to_sec()  # Convertir a segundos

    while self.path_planning_state != 2:
      rospy.sleep(0.1)

      if rospy.Time.now().to_sec() - init_time > 10:  # Restar en segundos
        rospy.logerr("Error en la trayectoria")
        # return 'abortar'
        break # DEBUG porq no se verifica bien la finalización de trayectorias
      
    rospy.loginfo("Trayectoria completada")
    
    return 'retirada'

def main():
  # Inicializar nodo ROS
  rospy.init_node('fsm_fusion_franka_vision')
  rospy.loginfo("DEBUG: Iniciando nodo")

  # Inicializar el VisionMonitor y el ButtonMonitor (Singleton)
  vision = VisionMonitor()
  buttons = ButtonMonitor()
  gripper = GripperManager()

  # Crear la FSM principal
  sm_fsm = StateMachine(outcomes=['succeeded', 'failed'])

  with sm_fsm:
    StateMachine.add('REPOSO', Reposo(buttons),
                      transitions={'buscar': 'BUSCANDO',
                                  'esperando': 'REPOSO'})

    StateMachine.add('BUSCANDO', Buscando(vision, buttons),
                      transitions={'encontrado': 'APROXIMANDO',
                                   'no_encontrado': 'BUSCANDO',
                                   'abortar': 'REPOSO'})
               
    StateMachine.add('APROXIMANDO', Aproximando(vision, buttons),
                      transitions={'aprox_completada': 'AGARRE',
                                  'abortar': 'REPOSO'},
                      remapping={'last_kp_R_Wrist': 'last_kp_R_Wrist',
                                'last_normal_vector': 'last_normal_vector'})

    StateMachine.add('AGARRE', Agarre(buttons, gripper),
                      transitions={'agarrado': 'RETIRADA',
                                  'abortar': 'REPOSO'},
                      remapping={'last_kp_R_Wrist': 'last_kp_R_Wrist',
                                'last_normal_vector': 'last_normal_vector'})
    
    StateMachine.add('RETIRADA', Retirada(),
                      transitions={'retirada': 'REPOSO',
                                  'abortar': 'REPOSO'},
                      remapping={'last_kp_R_Wrist': 'last_kp_R_Wrist',
                                'last_normal_vector': 'last_normal_vector'})

  # Servidor de introspección para visualizar en SMACH Viewer
  sis = smach_ros.IntrospectionServer('server_name', sm_fsm, '/SM_ROOT')
  sis.start()

  # Ejecutar la máquina de estados
  outcome = sm_fsm.execute()

  # Detener introspección cuando termine
  sis.stop()

if __name__ == '__main__':
    main()