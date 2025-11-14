#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R


class VectorVisualizer:
    def __init__(self, topic_name="/normal_vector_marker", frame_id="base_link"):
        """
        Inicializa un publicador de RViz para visualizar vectores como flechas.

        :param topic_name: Nombre del topic donde se publicará el marcador.
        :param frame_id: Frame de referencia para el vector.
        """
        self.marker_pub = rospy.Publisher(topic_name, Marker, queue_size=10)
        self.frame_id = frame_id

    def publish_vector(self, origin, vector, color=(0.0, 0.0, 1.0), scale=0.1):
        """
        Publica un vector en RViz como una flecha.

        :param origin: Lista [x, y, z] con la posición de origen del vector.
        :param vector: Lista [dx, dy, dz] con la dirección del vector.
        :param color: Tupla (r, g, b) para definir el color del vector.
        :param scale: Tamaño de la flecha (grosor y longitud ajustable).
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "vector_visualizer"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Origen del vector
        start_point = Point(*origin)

        # Punto final del vector (origen + dirección escalada)
        end_point = Point()
        end_point.x = origin[0] + vector[0] * scale
        end_point.y = origin[1] + vector[1] * scale
        end_point.z = origin[2] + vector[2] * scale

        marker.points.append(start_point)
        marker.points.append(end_point)

        # Escala del vector
        marker.scale.x = 0.02  # Grosor de la línea
        marker.scale.y = 0.05  # Tamaño de la cabeza de la flecha
        marker.scale.z = 0.05

        # Color del vector
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0  # Opacidad

        marker.lifetime = rospy.Duration(0)  # Dura indefinidamente

        self.marker_pub.publish(marker)
        rospy.logdebug(f"Vector publicado en RViz: Origen={origin}, Dirección={vector}, Color={color}")

class PointVisualizer:
    def __init__(self, topic_name="/point_marker", frame_id="base_link"):
        """
        Inicializa un publicador de RViz para visualizar puntos en 3D.

        :param topic_name: Nombre del topic donde se publicará el marcador.
        :param frame_id: Frame de referencia para el punto.
        """
        self.marker_pub = rospy.Publisher(topic_name, Marker, queue_size=10)
        self.frame_id = frame_id

    def publish_point(self, np_point, color=(0.0, 0.0, 1.0), scale=0.1):
        """
        Publica un punto en RViz.

        :param np_point: np.array con las coordenadas del punto a visualizar.
        :param color: Tupla (r, g, b) para definir el color del punto.
        :param scale: Escala del punto (puedes ajustar el tamaño).
        """
        if not isinstance(np_point, np.ndarray) or np_point.shape != (3,):
            raise ValueError("El parámetro np_point debe ser un np.array de 3 elementos.")

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "point_visualizer"
        marker.id = 0
        marker.type = Marker.SPHERE  # Usamos esfera para representar un punto
        marker.action = Marker.ADD

        # Establecemos la posición de la esfera
        marker.pose.position = Point(np_point[0], np_point[1], np_point[2])

        # Escala del punto (tamaño de la esfera)
        marker.scale.x = scale  # Radio de la esfera
        marker.scale.y = scale  # Radio de la esfera
        marker.scale.z = scale  # Radio de la esfera

        # Color del punto
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0  # Opacidad

        marker.lifetime = rospy.Duration(0)  # Dura indefinidamente

        # Si no quieres que haya rotación, establece la orientación como identidad.
        marker.pose.orientation.w = 1.0  # Quaternion identidad

        self.marker_pub.publish(marker)
        rospy.logdebug(f"Punto publicado en RViz: Coordenadas={np_point}, Color={color}")

class PoseStampedVisualizer:
    def __init__(self, topic_name="/pose_stamped_marker", frame_id="base_link"):
        """
        Inicializa un publicador de RViz para visualizar PoseStamped como una triada 3D.

        :param topic_name: Nombre del topic donde se publicará el marcador.
        :param frame_id: Frame de referencia para la pose.
        """
        self.marker_pub = rospy.Publisher(topic_name, Marker, queue_size=10)
        self.frame_id = frame_id

    def publish_pose(self, pose, scale=0.1):
        """
        Publica una triada 3D (ejes X, Y, Z) en RViz para la pose.

        :param pose: PoseStamped que contiene la posición y orientación a visualizar.
        :param scale: Escala de las flechas que representan los ejes.
        """
        # Crear los marcadores para los tres ejes (X, Y, Z)
        marker_x = Marker()
        marker_y = Marker()
        marker_z = Marker()

        # Configuración común de los marcadores
        for marker in [marker_x, marker_y, marker_z]:
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "pose_visualizer"
            marker.action = Marker.ADD
            marker.pose.position = pose.pose.position  # Usamos la misma posición para los tres ejes
            marker.lifetime = rospy.Duration(0)  # Dura indefinidamente

        # Configuración para el eje X (rojo)
        marker_x.id = 0
        marker_x.type = Marker.ARROW
        marker_x.scale.x = 0.01  # Grosor de la flecha
        marker_x.scale.y = 0.05  # Tamaño de la cabeza de la flecha
        marker_x.scale.z = 0.05  # Tamaño de la cabeza de la flecha
        marker_x.color.r = 1.0
        marker_x.color.g = 0.0
        marker_x.color.b = 0.0
        marker_x.color.a = 1.0
        marker_x.points = [Point(0, 0, 0), Point(scale, 0, 0)]  # Flecha apuntando a lo largo del eje X

        # Configuración para el eje Y (verde)
        marker_y.id = 1
        marker_y.type = Marker.ARROW
        marker_y.scale.x = 0.01  # Grosor de la flecha
        marker_y.scale.y = 0.05  # Tamaño de la cabeza de la flecha
        marker_y.scale.z = 0.05  # Tamaño de la cabeza de la flecha
        marker_y.color.r = 0.0
        marker_y.color.g = 1.0
        marker_y.color.b = 0.0
        marker_y.color.a = 1.0
        marker_y.points = [Point(0, 0, 0), Point(0, scale, 0)]  # Flecha apuntando a lo largo del eje Y

        # Configuración para el eje Z (azul)
        marker_z.id = 2
        marker_z.type = Marker.ARROW
        marker_z.scale.x = 0.01  # Grosor de la flecha
        marker_z.scale.y = 0.05  # Tamaño de la cabeza de la flecha
        marker_z.scale.z = 0.05  # Tamaño de la cabeza de la flecha
        marker_z.color.r = 0.0
        marker_z.color.g = 0.0
        marker_z.color.b = 1.0
        marker_z.color.a = 1.0
        marker_z.points = [Point(0, 0, 0), Point(0, 0, scale)]  # Flecha apuntando a lo largo del eje Z

        # Publicar los marcadores para los tres ejes
        self.marker_pub.publish(marker_x)
        self.marker_pub.publish(marker_y)
        self.marker_pub.publish(marker_z)

        rospy.logdebug(f"Triada publicada en RViz: Posición={pose.pose.position}, Orientación={pose.pose.orientation}")
