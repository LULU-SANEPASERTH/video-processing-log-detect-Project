import cv2
import csv
import os
import uuid
from datetime import datetime
from ultralytics import YOLO
from typing import Dict, Any, Set, List, Tuple

# =============================================================================
#  CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Paths ---
    "model_path": "runs/detect/train_V3.4/weights/best.pt",
    "video_path": "raw-videos/2025-07-22_08-02-14.mp4",
    "tracker_config_path": "ultralytics/cfg/trackers/botsort.yaml",
    "csv_log_path": "result.csv",

    # --- Measurement Parameters ---
    #original 0.44882
    "pixels_per_mm": 0.44882,
    "length_threshold_mm": 2600,
    "max_disappeared_frames": 35,

    # --- Detection & Display Parameters ---
    "yolo_img_size": 640,
    "yolo_confidence": 0.8,
    "exit_buffer_pixels": 1,
    "horizontal_line_y_position": 400,
    "display_width": 800,
    "display_height": 450,
}


class LogDetector:
    """
    A class to handle log detection, tracking, length measurement, and data logging
    from a video source using a YOLOv8 model.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the LogDetector with a given configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing all necessary settings.
        """
        self.config = config
        self.model = YOLO(self.config["model_path"])
        self.cap = cv2.VideoCapture(self.config["video_path"])

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.config['video_path']}")

        # --- Geometric & State Variables ---
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.exit_line = self.frame_width - self.config["exit_buffer_pixels"]
        self.horizontal_line = self.config["horizontal_line_y_position"]
        
        self.active_logs: Dict[int, Dict[str, Any]] = {}
        self.logged_tracker_ids: Set[int] = set()
        
        self._initialize_csv()
        self.last_count = self._get_last_count()

    def _initialize_csv(self):
        """Creates the CSV file and writes the header if it doesn't exist."""
        if not os.path.exists(self.config["csv_log_path"]):
            header = ['uuid', 'data', 'time', 'lenght_mm', 'count']
            with open(self.config["csv_log_path"], mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def _get_last_count(self) -> int:
        """Reads the last 'count' value from the CSV file to continue numbering."""
        if not os.path.exists(self.config["csv_log_path"]):
            return 0
        try:
            with open(self.config["csv_log_path"], mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                counts = [int(row['count']) for row in reader if row.get('count') and row['count'].isdigit()]
                return max(counts) if counts else 0
        except Exception as e:
            print(f"Warning: Could not read last count from CSV: {e}")
            return 0

    def _log_data(self, log_uuid: str, length_mm: float):
        """Appends a single record to the CSV log file."""
        self.last_count += 1
        now = datetime.now()
        row = [
            log_uuid,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            f"{length_mm:.2f}",
            self.last_count,
        ]
        with open(self.config["csv_log_path"], mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _process_detections(self, results: Any):
        """Processes all tracked objects in the current frame."""
        current_track_ids = set()
        if results[0].boxes.id is None:
            return current_track_ids

        boxes = results[0].boxes.xyxy.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        current_track_ids = set(track_ids)

        for box, track_id in zip(boxes, track_ids):
            if track_id in self.logged_tracker_ids:
                continue

            x1, y1, x2, y2 = box
            current_length_mm = (x2 - x1) / self.config["pixels_per_mm"]

            # --- Update or create log entry in memory ---
            if track_id not in self.active_logs:
                self.active_logs[track_id] = {
                    'uuid': str(uuid.uuid4()),
                    'best_length': current_length_mm,
                    'disappeared_frames': 0,
                    'box_coords': (x1, y1) # Store for text display
                }
            else:
                self.active_logs[track_id]['best_length'] = max(
                    self.active_logs[track_id]['best_length'], current_length_mm
                )
                self.active_logs[track_id]['disappeared_frames'] = 0
                self.active_logs[track_id]['box_coords'] = (x1, y1)
            
            # --- Check for exit condition to log data ---
            best_len = self.active_logs[track_id]['best_length']
            log_uuid = self.active_logs[track_id]['uuid']

            if (x2 >= self.exit_line and y2 >= self.horizontal_line and 
                best_len > self.config["length_threshold_mm"]):
                
                self._log_data(log_uuid, best_len)
                self.logged_tracker_ids.add(track_id)
                del self.active_logs[track_id]

        return current_track_ids

    def _handle_disappeared(self, current_track_ids: Set[int]):
        """Manages logs that are no longer in the current frame."""
        disappeared_ids = set(self.active_logs.keys()) - current_track_ids
        for track_id in list(disappeared_ids):
            self.active_logs[track_id]['disappeared_frames'] += 1
            if self.active_logs[track_id]['disappeared_frames'] > self.config["max_disappeared_frames"]:
                log_data = self.active_logs[track_id]
                log_uuid, best_len = log_data['uuid'], log_data['best_length']

                print(f"ID: {log_uuid[:8]} (TrackerID: {track_id}) disappeared. Logging best length: {best_len:.2f} mm")
                
                if best_len > self.config["length_threshold_mm"]:
                    self._log_data(log_uuid, best_len)
                
                self.logged_tracker_ids.add(track_id)
                del self.active_logs[track_id]
    
    def _draw_overlays(self, frame: Any):
        """Draws static lines and dynamic info text on the frame."""
        # Draw static lines
        cv2.line(frame, (self.exit_line, 0), (self.exit_line, self.frame_height), (0, 0, 255), 2)
        cv2.line(frame, (0, self.horizontal_line), (self.frame_width, self.horizontal_line), (255, 0, 0), 2)
        
        # Draw dynamic text for active logs
        for data in self.active_logs.values():
            log_uuid = data['uuid']
            display_len = data['best_length']
            x1, y1 = data['box_coords']
            text_to_display = f"ID: {log_uuid[:8]} L: {display_len:.2f} mm"
            cv2.putText(frame, text_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def run(self):
        """Main processing loop."""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("End of video or failed to read frame.")
                break

            # Perform object tracking
            results = self.model.track(
                frame,
                imgsz=self.config["yolo_img_size"],
                conf=self.config["yolo_confidence"],
                persist=True,
                tracker=self.config["tracker_config_path"],
                verbose=False
            )

            # Process the results
            current_track_ids = self._process_detections(results)
            self._handle_disappeared(current_track_ids)
            
            # Draw overlays for visualization
            annotated_frame = results[0].plot()
            self._draw_overlays(annotated_frame)

            # Display the frame
            resized_frame = cv2.resize(annotated_frame, (self.config["display_width"], self.config["display_height"]))
            cv2.imshow("Log Detection System", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, stopping.")
                break
        
        # --- Final cleanup after loop ends ---
        self._finalize_logging()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Processing finished. Data saved to {self.config['csv_log_path']}")

    def _finalize_logging(self):
        """Logs any remaining active logs when the process stops."""
        print(f"Finalizing... logging {len(self.active_logs)} remaining logs.")
        for data in list(self.active_logs.values()):
             if data['best_length'] > self.config["length_threshold_mm"]:
                self._log_data(data['uuid'], data['best_length'])


if __name__ == "__main__":
    try:
        detector = LogDetector(CONFIG)
        detector.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
