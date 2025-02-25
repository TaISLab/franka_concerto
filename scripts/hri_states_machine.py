#!/usr/bin/env python3
 
# Import the necessary libraries
import rospy # Python client library
from smach import State, StateMachine, Concurrence # State machine library
import smach_ros # Extensions for SMACH library to integrate it with ROS
from time import sleep # Handle time
import time
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from franka_buttons.msg import FrankaButtons
from std_msgs.msg import Int32
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP
from funciones_utiles import are_kp_valid, calculate_normal_vector, print_pose_named, calculate_gripper_position, calculate_quaternion_0_F


# Estado Monitor State
def button_cross_pressed(msg):
    if msg.cross:  # El botón cross se ha presionado
        rospy.logwarn("Botón CROSS detectado. Volviendo a REPOSO.")
        return False  # Interrumpe la ejecución del estado actual
    return True  # Continúa la ejecución normal  




class VisionMonitorState(State):
  """
  Este estado se encargada de leer los KP del sistema de visión y calcular los vectores de .
  """
  def __init__(self):
      State.__init__(self, 
                            outcomes=['keypoints_valid'],
                            output_keys=['kp_R_Wrist', 'kp_R_Elbow', 'kp_R_Shoulder', 'normal_vector', 'vect_forearm'])
      
      # Suscripción al sistema de visión
      self.sub = rospy.Subscriber('/skeleton_3D', Skeleton3D, self.obtain_skeleton3D_callback)

      # Variables para almacenar los KP
      self.kp_R_Wrist = np.array([0,0,0])
      self.kp_R_Elbow = np.array([0,0,0])
      self.kp_R_Shoulder = np.array([0,0,0])
      self.normal_vector = np.array([0,0,1])
      self.vect_forearm = np.array([0,0,0])

      self.kp_valid = False
      self.lost_time = None  # Tiempo en que se perdieron los KP

  def obtain_skeleton3D_callback(self, msg):
      """ Obtiene keypoints y calcula el vector normal. """
      # Extraigo los KP como arrays de 3 valores (x y z)
      keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])

      self.kp_R_Shoulder = keypointsX[6]
      self.kp_R_Elbow = keypointsX[8]
      self.kp_R_Wrist = keypointsX[10]

      if are_kp_valid(self.kp_R_Shoulder, self.kp_R_Elbow, self.kp_R_Wrist):
          # Calculo de vectores
          self.normal_vector = calculate_normal_vector(self.kp_R_Shoulder, self.kp_R_Elbow, self.kp_R_Wrist)
          self.vect_forearm = self.kp_R_Wrist - self.kp_R_Elbow

          self.kp_valid = True
          self.lost_time = None  # Reiniciar el contador de pérdida de KP
      else:
          rospy.logdebug("KP no válidos detectados. No se actualiza el valor.")
          self.kp_valid = False
          if self.lost_time is None:
              self.lost_time = rospy.Time.now()

  def execute(self, userdata):
    rospy.loginfo("Monitoreando keypoints en paralelo...")

    while not rospy.is_shutdown():
      if self.kp_valid:
        # Guardar KP en `userdata`
        userdata.kp_R_Wrist = self.kp_R_Wrist
        userdata.kp_R_Elbow = self.kp_R_Elbow
        userdata.kp_R_Shoulder = self.kp_R_Shoulder
        userdata.normal_vector = self.normal_vector
        userdata.vect_forearm = self.vect_forearm
        # return 'keypoints_valid'  # Continúa con el flujo normal
      
      elif self.lost_time and (rospy.Time.now() - self.lost_time).to_sec() > 3.0:
        rospy.logwarn("Los KP se han perdido por más de 3 segundos. Reiniciando búsqueda...")
        self.lost_time = rospy.Time.now()  # Reiniciar el contador en vez de salir
        # continue  # Mantener el estado activo y seguir buscando

      rospy.sleep(0.1)  # Ciclo de espera

  

# Definir estado REPOSO
class Reposo(State):
  def __init__(self):
    State.__init__(self, outcomes=['buscar','esperando', 'shutdown'])

    rospy.Subscriber("/franka_buttons", FrankaButtons, self.check_button_callback)
    
    self.check_button_pressed = False

  def check_button_callback(self, msg):
    """ Función que verifica si se ha pulsado el botón check """
    if msg.check: # Si se ha pulsado, true
      rospy.loginfo("Rep: Botón Check pulsado")
      self.check_button_pressed = True

  def execute(self, userdata):
    rospy.loginfo('Ejecutando estado de REPOSO...')
    
    while not rospy.is_shutdown():
        if self.check_button_pressed:
            self.check_button_pressed = False
            return 'buscar'
        rospy.sleep(0.1)  
    
    rospy.logwarn("Ctrl+C detectado. Terminando FSM.")
    return 'shutdown'  # Salir correctamente
    
# Estado BUSCANDO. El sistema busca que la muñeca esté dentro del espacio de trabajo
class Buscando(State):
  def __init__(self):
    State.__init__(self, outcomes=['encontrado','no_encontrado', 'shutdown'], input_keys=['kp_R_Wrist'])
 
    self.keypoint_detectado = None  # Almacena la posición del keypoint
    self.tiempo_inicio = None  # Guarda el tiempo en que el keypoint entra en la zona

  def esta_dentro_del_espacio(self, punto):
    """ Verifica si el punto está dentro del volumen cartesiano definido """
    x_min, x_max = 0.0, 0.55
    y_min, y_max = -0.3, 0.3
    z_min, z_max = 0.0, 0.8

    return (x_min <= punto.x <= x_max and 
            y_min <= punto.y <= y_max and 
            z_min <= punto.z <= z_max)

  def execute(self, userdata):
    rospy.loginfo("Ejecutando estado BUSCANDO...")
    ################################
    # IMPORTANTE!! Modificar para que estudie que la muñeca permanece quieta en el espacio
    ###############################

    # while not rospy.is_shutdown():
    #   if self.esta_dentro_del_espacio(userdata.kp_R_Wrist):
    #     if self.tiempo_inicio is None:  
    #         self.tiempo_inicio = time.time()  # inicio del contador
          
    #     if time.time() - self.tiempo_inicio >= 2.0:
    #         rospy.loginfo("KP ha permanecido 3 segundos en el espacio de busqueda.")
    #         return 'encontrado'  # al siguiente estado
    #     else:
    #       self.tiempo_inicio = None  # Se reinicia el contador si el kp sale del espacio

    #   rospy.sleep(0.1)  # pausa
      # return 'no_encontrado'
    rospy.sleep(2)  # pausa
    if self.esta_dentro_del_espacio(userdata.kp_R_Wrist):
       return 'encontrado'
    
    rospy.logwarn("Ctrl+C detectado. Terminando FSM.")
    return 'shutdown'  # Salir correctamente

class Aproximando(State):
  def __init__(self):
    State.__init__(self, outcomes=['aproximado','error', 'shutdown'], input_keys=['kp_R_Wrist', 'normal_vector', 'vect_forearm'])

    # subscripción al current_pose
    self.current_pose_subscriber = rospy.Subscriber("/current_pose", PoseStamped, self.current_pose_callback)
    # subscripción al estado de la planificación
    self.path_planning_state_subscriber = rospy.Subscriber("/path_planning_state", Int32, self.path_planning_state_callback)

    # publica en /desired_pose, entrada del planificador de trayectorias
    self.desired_pose_publisher = rospy.Publisher("/desired_pose", PoseStamped, queue_size=10)

    self.current_pose = PoseStamped()
    self.path_planning_state = -1

  def current_pose_callback(self, msg):
    rospy.logdebug("Pose actual recibida")
    self.current_pose = msg

  def path_planning_state_callback(self, msg):
    rospy.logdebug("Estado de la trayectoria recibido")
    self.path_planning_state = msg.data
  
  def execute(self, userdata):
    try:
      rospy.loginfo("Ejecutando estado Aproximando")
      distancia_aproximacion = 0.1

      desired_pose = PoseStamped()
      desired_pose.pose.position = calculate_gripper_position(userdata.kp_R_Wrist, userdata.normal_vector, distancia_aproximacion)
      desired_pose.pose.orientation = calculate_quaternion_0_F(userdata.vect_forearm, -userdata.normal_vector)
      self.desired_pose_publisher.publish(desired_pose) # publicar el kp q le pasa el estado anterior

      while (self.path_planning_state != 1):
        rospy.loginfo("Ejecutando trayectoria...")
        rospy.sleep(0.1)
        
      rospy.loginfo("Trayectoria completada")
      return "aproximado"
    
    except:
      return "error"


def main():
  # Initialize the node
  rospy.init_node('fsm_fusion_franka_vision')
  rospy.loginfo("DEBUG: Iniciando nodo")

  # Crear una concurrencia donde `VISION_MONITOR` y `MONITOR_CROSS` corren siempre
  sm_concurrente = Concurrence(
      outcomes=['shutdown'],
      default_outcome='shutdown',
      output_keys=['kp_R_Wrist', 'kp_R_Elbow', 'kp_R_Shoulder', 'normal_vector', 'vect_forearm'],
      child_termination_cb=lambda so: False,  # No termina si uno de los estados finaliza
      outcome_map={
          'shutdown': {
              'VISION_MONITOR': 'keypoints_valid',
              'MONITOR_CROSS': 'invalid',
              'FSM_PRINCIPAL': 'shutdown'  # Ahora FSM_PRINCIPAL sabe cómo salir
          }
      }
  )


  with sm_concurrente:
    # Estado concurrente 1: Monitor de visión (siempre corriendo)
    Concurrence.add('VISION_MONITOR', VisionMonitorState())
    
    # Estado concurrente 2: Monitor del botón CROSS (siempre corriendo)
    Concurrence.add('MONITOR_CROSS',
        smach_ros.MonitorState('/franka_buttons', FrankaButtons, button_cross_pressed,
                              input_keys=['cross'])  # Agregar input_keys
    )

    
    
    # Crear la FSM principal dentro de la concurrencia (correrá en paralelo con los monitores)
    sm_fsm = StateMachine(outcomes=['succeeded', 'failed', 'shutdown'])  # Agregamos shutdown

    
    # Set user data for the finite state machine
    sm_fsm.userdata.kp_R_Wrist = Point(0, 0, 0)
    sm_fsm.userdata.kp_R_Elbow = Point(0, 0, 0)
    sm_fsm.userdata.kp_R_Shoulder = Point(0, 0, 0)
    sm_fsm.userdata.normal_vector = np.array([0, 0, 1])
    sm_fsm.userdata.vect_forearm = np.array([0, 0, 0])

    with sm_fsm:
      StateMachine.add('REPOSO', Reposo(),
                       transitions={'buscar': 'BUSCANDO',
                                'esperando': 'REPOSO',
                                'shutdown': 'shutdown'},
                       remapping={'kp_R_Wrist': 'kp_R_Wrist'})
      StateMachine.add('BUSCANDO', Buscando(),
                      transitions={'encontrado': 'APROXIMANDO',
                                    'no_encontrado': 'BUSCANDO',
                                    'shutdown': 'shutdown'},
                       remapping={'kp_R_Wrist': 'kp_R_Wrist'})

      StateMachine.add('APROXIMANDO', Aproximando(),
                      transitions={'aproximado': 'succeeded',
                                    'error': 'failed',
                                    'shutdown': 'shutdown'},
                       remapping={'kp_R_Wrist': 'kp_R_Wrist',
                                  'normal_vector': 'normal_vector',
                                  'vect_forearm': 'vect_forearm'})

    # Agregar la FSM a la concurrencia
    Concurrence.add('FSM_PRINCIPAL', sm_fsm)
  
  # Instrospection server: necessary to view our state transitions using ROS
  sis = smach_ros.IntrospectionServer('server_name', sm_concurrente, '/SM_ROOT')
  sis.start()
  
  try:
    outcome = sm_concurrente.execute()
  except KeyboardInterrupt:
      rospy.logwarn("Ctrl+C detectado. Terminando ejecución...")
      sm_concurrente.request_preempt()  # Pedir que la concurrencia termine
      rospy.signal_shutdown("Nodo finalizado por usuario")  # Apagar ROS correctamente
  finally:
      sis.stop()  # Detener introspección de SMACH


if __name__ == '__main__':
    main()