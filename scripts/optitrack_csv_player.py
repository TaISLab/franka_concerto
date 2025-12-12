#!/usr/bin/env python3
import rospy
import csv
import sys
import os
import bisect
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

# --- DEFINICIÓN DE JOINTS ---
JOINTS_MAP = {
    'elbow':     (1, 2, 3),
    'lshoulder': (4, 5, 6),
    'rshoulder': (7, 8, 9),
    'stick':     (10, 11, 12),
    'wrist':     (13, 14, 15)
}

class OptitrackPlayer:
    def __init__(self):
        rospy.init_node('optitrack_csv_player', anonymous=True)

        # 1. Verificar use_sim_time (Vital para sincronizar con bag externo)
        if not rospy.get_param("/use_sim_time", False):
            rospy.logwarn("⚠️ ATENCIÓN: /use_sim_time es False. Si vas a usar un rosbag, ponlo a True.")

        # 2. Cargar configuración desde el Parameter Server
        # (Se asume que cargaste el YAML con el launch anterior)
        ns = "/exp_optitrack_25" 
        
        # Leemos los parámetros necesarios para sincronizar
        self.csv_path = rospy.get_param(f"{ns}/optitrack_csv_path", "")
        self.bag_start_ts = rospy.get_param(f"{ns}/rosbag_start_time", 0.0)
        self.time_offset = rospy.get_param(f"{ns}/delay_optitrack_sec", 0.0)
        
        if not self.csv_path or not os.path.exists(self.csv_path):
            rospy.logerr(f"❌ No se encuentra el CSV en: {self.csv_path}")
            sys.exit(1)

        # 3. Configurar Publishers
        self.point_pubs = {name: rospy.Publisher(f'/optitrack/{name}', PointStamped, queue_size=1) for name in JOINTS_MAP}
        self.marker_pub = rospy.Publisher('/optitrack/markers', MarkerArray, queue_size=1)
        
        # 4. Cargar TODO el CSV en Memoria (RAM) para permitir saltos temporales
        self.timestamps = []
        self.data_rows = []
        self.load_csv()

    def load_csv(self):
        rospy.loginfo(f"📥 Cargando CSV en memoria: {self.csv_path}")
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader) # Saltar cabecera si existe
                for row in reader:
                    try:
                        # Asumimos que la columna 0 es el tiempo relativo del experimento
                        t = float(row[0])
                        self.timestamps.append(t)
                        self.data_rows.append(row)
                    except ValueError:
                        continue
            rospy.loginfo(f"✅ CSV cargado: {len(self.timestamps)} fotogramas listos.")
        except Exception as e:
            rospy.logerr(f"Error leyendo CSV: {e}")
            sys.exit(1)

    def get_frame_index_at_time(self, rel_time):
        """
        Busca el índice más cercano al tiempo dado usando Bisección.
        Esto permite saltos (seeking) y rebobinado eficiente.
        """
        # bisect_left encuentra el punto de inserción para mantener el orden
        idx = bisect.bisect_left(self.timestamps, rel_time)
        
        # Ajustes de límites
        if idx >= len(self.timestamps):
            return len(self.timestamps) - 1
        if idx < 0:
            return 0
        return idx

    def run(self):
        rate = rospy.Rate(100) # Frecuencia de actualización de visualización
        
        rospy.loginfo("⏳ Esperando reloj de simulación (/clock)...")
        # Bloqueo hasta que empiece el bag
        while not rospy.Time.now().to_sec() > 0 and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # CALCULO DEL TIEMPO CERO ABSOLUTO
        # El t=0.0 del CSV corresponde en el mundo ROS a:
        # T_global = (Timestamp inicio bag) + (Offset definido en yaml)
        bag_start_time_ros = rospy.Time(self.bag_start_ts)
        global_start_opti = bag_start_time_ros + rospy.Duration(self.time_offset)

        rospy.loginfo("▶️ Sincronización activa. Escuchando tiempo del Bag...")

        while not rospy.is_shutdown():
            # 1. ¿Qué hora es en el Bag?
            sim_now = rospy.Time.now()
            
            # 2. ¿En qué segundo del CSV estaríamos?
            time_in_csv = (sim_now - global_start_opti).to_sec()

            # 3. Buscar el fotograma (soporta t negativo o saltos)
            idx = self.get_frame_index_at_time(time_in_csv)
            
            # Imprimir info en una sola línea (retorno de carro \r)
            sys.stdout.write(f"\r[BAG: {sim_now.to_sec():.2f}] | [CSV T: {time_in_csv:.2f}s] | [Frame: {idx}/{len(self.timestamps)}]")
            sys.stdout.flush()

            # Si el bag está antes de que empezara el optitrack (tiempo negativo), no publicar nada
            if time_in_csv < 0:
                rate.sleep()
                continue
            
            # 4. Publicar datos
            row = self.data_rows[idx]
            self.publish_data(row, sim_now)

            rate.sleep()
            
        print("\n🛑 Nodo detenido.")

    def publish_data(self, row, stamp):
        marker_array = MarkerArray()
        id_counter = 0
        frame_id = 'base_link' # Asegúrate que coincide con tu TF tree

        for joint_name, indices in JOINTS_MAP.items():
            idx_x, idx_y, idx_z = indices
            try:
                x, y, z = float(row[idx_x]), float(row[idx_y]), float(row[idx_z])
            except (IndexError, ValueError): 
                continue

            # PointStamped (Para plots)
            ps = PointStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = frame_id
            ps.point.x, ps.point.y, ps.point.z = x, y, z
            self.point_pubs[joint_name].publish(ps)

            # Marker (Para RViz 3D)
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = "optitrack_skeleton"
            marker.id = id_counter
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05; marker.scale.y = 0.05; marker.scale.z = 0.05
            marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
            marker.lifetime = rospy.Duration(0.02) # Efímero para evitar estelas al saltar
            marker_array.markers.append(marker)
            id_counter += 1

        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        player = OptitrackPlayer()
        player.run()
    except rospy.ROSInterruptException:
        pass