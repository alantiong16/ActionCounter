import random
from pythonosc import udp_client
import numpy as np
# OSC server address and port
osc_server_address = "127.0.0.1"  # Replace with your OSC server address
osc_server_port = 12345  # Replace with your OSC server port

# Create an OSC client
osc_client = udp_client.SimpleUDPClient(osc_server_address, osc_server_port)

KINECT_NODES = {'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3, 'CLAVICLE_LEFT': 4, 'SHOULDER_LEFT': 5, 'ELBOW_LEFT': 6, 'WRIST_LEFT': 7, 'HAND_LEFT': 8, 'HANDTIP_LEFT': 9, 'THUMB_LEFT': 10, 'CLAVICLE_RIGHT': 11, 'SHOULDER_RIGHT': 12, 'ELBOW_RIGHT': 13, 'WRIST_RIGHT': 14, 'HAND_RIGHT': 15, 'HANDTIP_RIGHT': 16, 'THUMB_RIGHT': 17, 'HIP_LEFT': 18, 'KNEE_LEFT': 19, 'ANKLE_LEFT': 20, 'FOOT_LEFT': 21, 'HIP_RIGHT': 22, 'KNEE_RIGHT': 23, 'ANKLE_RIGHT': 24, 'FOOT_RIGHT': 25, 'HEAD': 26, 'NOSE': 27, 'EYE_LEFT': 28, 'EAR_LEFT': 29, 'EYE_RIGHT': 30, 'EAR_RIGHT': 31}

def send_skel():
    # Generate a random float between 0.0 and 1.0
    # random_float = random.uniform(0.0, 1.0)
    # random_float =  np.random.randn(3)
    skel_json = {}
    json_string=''
    for NODE in KINECT_NODES:
        # skel_json[NODE] = str(np.random.randn(3).tolist())
        # skel_json[NODE] = str(np.random.randn(3).tolist())
        rng = np.random.randn(3)
        json_string+=f'{NODE}:{rng[0]},{rng[1]},{rng[2]}_'
    # Send the random float to the OSC server
    osc_client.send_message("/dummy_json", json_string)
    print(f"Sent dummy json: {json_string}")

from KinectLib import KinectLib
if __name__ == "__main__":
    # You can run this loop to send random floats continuously
    kin = KinectLib()
    while True:
        payload = kin.get_skel_pos_str()
        print(osc_client.send_message("/dummy_json", payload))
    kin.device.close()
        # send_skel(payload)

