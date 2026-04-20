from typing import *

import cv2
from ultralytics import YOLO
import json

# Load a pretrained YOLO model
model = YOLO("yolo26n.pt")

def train():
    # Train the model
    model.train(data="platform_dataset/data.yaml", epochs=200, batch=4, workers=0, device="cuda:0")
    
    new_model = YOLO(model="runs/detect/train/weights/best.pt")
    results = new_model("test_img.jpg", device="cuda:0")
    results[0].show()

def write_tray_pos(x:float, y:float, block_id: int, name:str = "substrate_trays"):
    """
    :param x: (left top x of the detected tray object - left top x of the current block)/width of the current block
    :param y: (left top y of the detected tray object - left top y of the current block)/length of the current block
    :param block_id: Which block the tray is located in
    :param name: What kind of tray. We have substrate_trays and bottle_trays
    """
    with open('./tray.json', 'r') as file:
        tray_data = json.load(file)

    block_conf = read_layout_block(block_id)
    x0 = block_conf['x']
    y0 = block_conf['y']
    w = block_conf['w']
    h = block_conf['h']

    tray_data[name][block_id - 1]['position']['x'] = int(x0+x*w)
    tray_data[name][block_id - 1]['position']['y'] = int(y0+y*h)

    # Write the updated data back to the file
    with open('tray.json', 'w') as file:
        json.dump(tray_data, file, indent=2)

def read_layout_block(block_id: int):
    with open('blocks.json', 'r') as file:
        layout = json.load(file)
    return layout['blocks'][block_id]

#start tracking
def run_scan(block_id:int = 1) -> Optional[Dict[int, Dict]]:
    scan_result = {"bottle_trays": [], "substrate_trays": []}

    model = YOLO(model="runs/detect/train_tray_and_block/weights/best.pt")
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Set camera resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Verify the resolution was set correctly
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}×{actual_height}")
    frame_center_x = actual_width / 2
    frame_center_y = actual_height / 2

    for i in range(500):# Start recording for 500 loops, each frame has 1ms delay
        # Read frame from camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Ensure frame is 640x480 (resize if needed)
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))

        # Run tracking
        results = model.track(frame, device="cpu", conf=0.3)

        # Display the frame
        cv2.imshow("Camera", results[0].plot())


        if i == 499:
            # Filter boxes, only the most centered layout block and trays in it are reserved
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()  # Class IDs; 0: bottle_tray, 1: layout_block, 2: substrate_tray

                indices_to_keep = []
                min_distance = float('inf')
                center_block_idx = -1
                center_block_pos = None
                for i,class_id in enumerate(classes):
                    if class_id == 1:
                        x1, y1, x2, y2 = boxes[i]
                        box_center_x = (x1 + x2) / 2
                        box_center_y = (y1 + y2) / 2
                        distance = ((box_center_x - frame_center_x) ** 2 +
                                    (box_center_y - frame_center_y) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            center_block_idx = i
                            center_block_pos = boxes[i]
                if center_block_idx != -1:
                    indices_to_keep.append(center_block_idx)
                for i,class_id in enumerate(classes):
                    if class_id != 1 and center_block_pos is not None:
                        x1, y1, x2, y2 = boxes[i]
                        x3, y3, x4, y4 = center_block_pos
                        if x1 > x3 and y1 > y3 and x2 < x4 and y2 < y4:
                            indices_to_keep.append(i)
                            block_wid = center_block_pos[2] - center_block_pos[0]
                            block_len = center_block_pos[3] - center_block_pos[1]
                            pos_norm = [
                                (boxes[i][0] - center_block_pos[0])/block_wid,
                                (boxes[i][1] - center_block_pos[1])/block_len
                            ]
                            if class_id == 0:
                                scan_result["bottle_trays"].append(pos_norm)
                            elif class_id == 2:
                                scan_result["substrate_trays"].append(pos_norm)

            results[0].boxes = results[0].boxes[indices_to_keep]

        # Wait for key press (1ms delay for real-time display)
        key = cv2.waitKey(1) & 0xFF

        # Check if 'q' is pressed to quit
        if key == ord('q'):
            print("Quitting...")
            break
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    return {block_id: scan_result}

# block_id = 1
# pos_conf = run_scan(block_id)
# results = pos_conf[block_id]
# substrate_pos = results["substrate_trays"]
# bottle_pos = results["bottle_trays"]
# for pos in substrate_pos:
#     write_tray_pos(float(pos[0]), float(pos[1]), block_id)
# for pos in bottle_pos:
#     write_tray_pos(float(pos[0]), float(pos[1]), block_id, "bottle_trays")

