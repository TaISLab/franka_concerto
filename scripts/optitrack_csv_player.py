#!/usr/bin/env python3
import rospy
import csv
import sys
import os
import bisect
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

class OptitrackPlayer:
    def __init__(self):
        rospy.init_node('optitrack_csv_player', anonymous=True)

        # 1. Verificar use_sim_time
        if not rospy.get_param("/use_sim_time", False):
            rospy.logwarn("⚠️ ATENCIÓN: /use_sim_time es False. Si vas a usar un rosbag, ponlo a True.")

        # 2. Configuración
        ns = "/exp_optitrack_25"
        self.csv_path = rospy.get_param(f"{ns}/optitrack_csv_path", "")
        self.bag_start_ts = rospy.get_param(f"{ns}/rosbag_start_time", 0.0)
        self.time_offset = rospy.get_param(f"{ns}/delay_optitrack_sec", 0.0)
        
        # Leer configuración de qué puntos queremos visualizar
        self.optipoints = rospy.get_param(f"{ns}/optipoints", {})

        if not self.csv_path or not os.path.exists(self.csv_path):
            rospy.logerr(f"❌ No se encuentra el CSV en: {self.csv_path}")
            sys.exit(1)

        # 3. Cargar CSV y construir el mapa dinámicamente
        self.timestamps = []
        self.data_rows = []
        self.joints_map = {}
        
        # Esta función lee el CSV, rellena los datos y crea el joints_map mirando la cabecera
        self.load_csv_and_build_map()

        # 4. Configurar Publishers (Solo para los joints encontrados)
        rospy.loginfo(f"📡 Creando publishers para {len(self.joints_map)} articulaciones encontradas.")
        self.point_pubs = {name: rospy.Publisher(f'/optitrack/{name}', PointStamped, queue_size=1) for name in self.joints_map}
        self.marker_pub = rospy.Publisher('/optitrack/markers', MarkerArray, queue_size=1)
        
    def load_csv_and_build_map(self):
        rospy.loginfo(f"📥 Cargando y analizando CSV: {self.csv_path}")
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                
                # A) LEER Y PROCESAR CABECERA
                header_row = next(reader)
                # Limpiar espacios en blanco (ej: "Time  " -> "Time")
                header = [h.strip() for h in header_row]
                
                # Construir mapa de índices
                self.joints_map = self.find_indices_from_header(header)

                # B) LEER DATOS
                for row in reader:
                    try:
                        # Asumimos que la columna 0 siempre es el Time
                        t = float(row[0])
                        self.timestamps.append(t)
                        self.data_rows.append(row)
                    except ValueError:
                        continue
            
            rospy.loginfo(f"✅ CSV cargado: {len(self.timestamps)} fotogramas.")
            
        except Exception as e:
            rospy.logerr(f"Error crítico leyendo CSV: {e}")
            sys.exit(1)

    def find_indices_from_header(self, header):
        """
        Busca dinámicamente las columnas _X, _Y, _Z para cada nombre de joint deseado.
        """
        joints_indices = {}
        
        # Lista de candidatos base que siempre queremos buscar
        candidate_joints = [
            'elbow', 'lshoulder', 'rshoulder', 'stick', 'wrist',
        ]

        # Añadir candidatos según parámetros (YAML)
        if self.optipoints.get('primitives', False):
            candidate_joints.extend([
                'elbow_1', 'elbow_2',
                'lshoulder_1', 'lshoulder_2',
                'rshoulder_1', 'rshoulder_2',
                'stick_1', 'stick_2'
            ])

        if self.optipoints.get('new_wrist', False):
            candidate_joints.append('wrist_mod')

        if self.optipoints.get('shoulder_correction', True):
            candidate_joints.extend([
                'rshoulder_c', 'lshoulder_c', 
                'rshoulder_1_c', 'rshoulder_2_c', # Por si acaso están en el CSV
                'lshoulder_1_c', 'lshoulder_2_c'
            ])

        # Búsqueda en la cabecera
        for joint in candidate_joints:
            try:
                # Intentamos encontrar las 3 columnas
                col_x = f"{joint}_X"
                col_y = f"{joint}_Y"
                col_z = f"{joint}_Z"

                if col_x in header and col_y in header and col_z in header:
                    idx_x = header.index(col_x)
                    idx_y = header.index(col_y)
                    idx_z = header.index(col_z)
                    
                    # Guardamos la tupla
                    joints_indices[joint] = (idx_x, idx_y, idx_z)
                    rospy.logdebug(f"🔹 Mapped {joint}: indices ({idx_x}, {idx_y}, {idx_z})")
                else:
                    # Si falta alguna coordenada, no lo añadimos
                    pass
            except ValueError:
                continue

        return joints_indices

    def get_frame_index_at_time(self, rel_time):
        idx = bisect.bisect_left(self.timestamps, rel_time)
        if idx >= len(self.timestamps):
            return len(self.timestamps) - 1
        if idx < 0:
            return 0
        return idx

    def run(self):
        rate = rospy.Rate(100)
        
        rospy.loginfo("⏳ Esperando reloj de simulación (/clock)...")
        while not rospy.Time.now().to_sec() > 0 and not rospy.is_shutdown():
            rospy.sleep(0.1)

        bag_start_time_ros = rospy.Time(self.bag_start_ts)
        global_start_opti = bag_start_time_ros + rospy.Duration(self.time_offset)

        rospy.loginfo("▶️ Sincronización activa.")

        while not rospy.is_shutdown():
            sim_now = rospy.Time.now()
            time_in_csv = (sim_now - global_start_opti).to_sec()
            idx = self.get_frame_index_at_time(time_in_csv)
            
            # Feedback visual
            sys.stdout.write(f"\r[BAG: {sim_now.to_sec():.2f}] | [CSV T: {time_in_csv:.2f}s] | [Frame: {idx}/{len(self.timestamps)}]")
            sys.stdout.flush()

            if time_in_csv < 0:
                rate.sleep()
                continue
            
            row = self.data_rows[idx]
            self.publish_data(row, sim_now)

            rate.sleep()
        print("\n🛑 Nodo detenido.")

    def publish_data(self, row, stamp):
        marker_array = MarkerArray()
        id_counter = 0
        frame_id = 'base_link'

        for joint_name, indices in self.joints_map.items():
            idx_x, idx_y, idx_z = indices
            try:
                # Extraer datos usando los índices dinámicos
                x = float(row[idx_x])
                y = float(row[idx_y])
                z = float(row[idx_z])
            except (IndexError, ValueError): 
                continue

            # PointStamped
            ps = PointStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = frame_id
            ps.point.x, ps.point.y, ps.point.z = x, y, z
            self.point_pubs[joint_name].publish(ps)

            # Marker
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
            marker.scale.x = 0.02; marker.scale.y = 0.02; marker.scale.z = 0.02
            marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
            marker.lifetime = rospy.Duration(0.02)
            marker_array.markers.append(marker)
            id_counter += 1

        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        player = OptitrackPlayer()
        player.run()
    except rospy.ROSInterruptException:
        pass