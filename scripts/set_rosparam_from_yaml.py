import sys
import os
import yaml
import rospy

def set_rosparams_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    for key, value in params.items():
        rospy.set_param('/' + key, value)
    print("Todos los parámetros han sido cargados como parámetros globales de ROS.")

def delete_rosparams_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    for key in params.keys():
        try:
            rospy.delete_param('/' + key)
            print(f"Parámetro eliminado: /{key}")
        except KeyError:
            print(f"Parámetro no encontrado (no eliminado): /{key}")
    print("Todos los parámetros han sido eliminados de los parámetros globales de ROS.")

if __name__ == "__main__":
    rospy.init_node('yaml_to_rosparam', anonymous=True)
    if len(sys.argv) < 2:
        print("Uso: python set_rosparams_from_yaml.py <path_to_yaml> [delete]")
        sys.exit(1)
    yaml_path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "delete":
        print(f"Eliminando parámetros definidos en: {os.path.basename(yaml_path)}")
        delete_rosparams_from_yaml(yaml_path)
    else:
        print(f"Cargando parámetros desde: {os.path.basename(yaml_path)}")
        set_rosparams_from_yaml(yaml_path)