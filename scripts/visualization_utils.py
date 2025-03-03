#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

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

