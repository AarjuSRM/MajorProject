import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QMessageBox,
                           QScrollArea)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
import os
from datetime import datetime

class YOLOThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(str)
    violation_signal = pyqtSignal(str, np.ndarray, int, object)  # Added results to signal

    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        self.file_path = None
        self.processed_frames = []
        self.current_frame = None
        self.violations_dir = "helmet_violations"
        self.bbox_dir = "helmet_violations/bbox_data"  # New directory for bbox data
        self.is_video = False
        self.total_frames = 0
        self.violation_frames = set()
        
        # Create necessary directories
        for directory in [self.violations_dir, self.bbox_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def save_bbox_data(self, results, filename_base):
        # Get the txt file path
        txt_path = os.path.join(self.bbox_dir, f"{filename_base}.txt")
        
        # Open file to write bbox data
        with open(txt_path, 'w') as f:
            boxes = results[0].boxes
            for box in boxes:
                # Get class, confidence and normalized coordinates
                cls = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                
                # Save all detections with class label
                # Format: class_id x_center y_center width height confidence
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

    def check_violation(self, results, frame, frame_number):
        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        
        RIDER_CLASS = 2  # rider
        NO_HELMET_CLASS = 1  # without helmet
        
        if (RIDER_CLASS in classes) and (NO_HELMET_CLASS in classes):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.is_video:
                message = f"Violation in Frame {frame_number}/{self.total_frames} at {timestamp}:\n"
            else:
                message = f"Violation detected at {timestamp}:\n"
            
            message += f"Rider detected with confidence: {conf[classes == RIDER_CLASS].max():.2f}\n"
            message += f"No helmet detected with confidence: {conf[classes == NO_HELMET_CLASS].max():.2f}"
            
            self.violation_frames.add(frame_number)
            
            # Pass the results object along with other data
            self.violation_signal.emit(message, frame, frame_number, results)

    def run(self):
        self.violation_frames.clear()
        
        if self.file_path.lower().endswith(('.mp4', '.avi')):
            self.is_video = True
            self.processed_frames = []
            cap = cv2.VideoCapture(self.file_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_number += 1
                    self.progress_signal.emit(f"Processing frame {frame_number}/{self.total_frames}")
                    
                    results = self.model(frame)
                    result_frame = results[0].plot()
                    self.processed_frames.append(result_frame)
                    self.change_pixmap_signal.emit(result_frame)
                    
                    self.check_violation(results, frame, frame_number)
                else:
                    break
            cap.release()
            
            self.progress_signal.emit(
                f"Processing complete. Found violations in {len(self.violation_frames)} "
                f"out of {self.total_frames} frames."
            )
        else:
            self.is_video = False
            image = cv2.imread(self.file_path)
            results = self.model(image)
            result_frame = results[0].plot()
            self.current_frame = result_frame
            self.change_pixmap_signal.emit(result_frame)
            
            self.check_violation(results, image, 1)
        
        self.finished_signal.emit()

    def set_file(self, file_path):
        self.file_path = file_path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Helmet Detection")
        self.setGeometry(100, 100, 1200, 900)  # Increased window size

        # Create central widget and layouts
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        # Create status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Create violation log with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(150)
        scroll_area.setMaximumHeight(200)
        
        self.violation_log = QLabel("Violation Log:")
        self.violation_log.setAlignment(Qt.AlignTop)
        self.violation_log.setStyleSheet("background-color: white; border: 1px solid gray; padding: 5px;")
        self.violation_log.setWordWrap(True)
        
        scroll_area.setWidget(self.violation_log)
        layout.addWidget(scroll_area)
        
        # Create image display label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        layout.addWidget(self.image_label)

        # Create button layout
        button_layout = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load Image/Video", self)
        self.load_button.clicked.connect(self.load_file)
        button_layout.addWidget(self.load_button)

        # Process button
        self.process_button = QPushButton("Process", self)
        self.process_button.clicked.connect(self.process_file)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.process_button)

        # Save button
        self.save_button = QPushButton("Save Output", self)
        self.save_button.clicked.connect(self.save_output)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        # Add button layout to main layout
        layout.addLayout(button_layout)
        central_widget.setLayout(layout)

        # Initialize variables
        self.file_path = None
        self.yolo_thread = YOLOThread('/Users/aarjukumar/Aarju_Kumar/Development/BikeHelmetDetection_FaceDetection/YOLOv9t_best.pt')
        self.yolo_thread.change_pixmap_signal.connect(self.update_image)
        self.yolo_thread.finished_signal.connect(self.processing_finished)
        self.yolo_thread.progress_signal.connect(self.update_status)
        self.yolo_thread.violation_signal.connect(self.handle_violation)

    def handle_violation(self, message, frame, frame_number, results):
        # Update violation log
        current_text = self.violation_log.text()
        if current_text == "Violation Log:":
            current_text = ""
        self.violation_log.setText(current_text + "\n" + message)
        
        # Create base filename
        base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        if self.yolo_thread.is_video:
            filename_base = f"{base_filename}_frame_{frame_number:04d}"
        else:
            filename_base = f"{base_filename}"
        
        # Save violation image
        image_path = os.path.join(self.yolo_thread.violations_dir, f"{filename_base}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Save bounding box data
        self.yolo_thread.save_bbox_data(results, filename_base)
        
        # Print to console
        print(f"\nViolation detected:")
        print(f"Image saved to: {image_path}")
        print(f"Bbox data saved to: {os.path.join(self.yolo_thread.bbox_dir, filename_base + '.txt')}")
        print(message)

    def load_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(
            self, 
            "Open Image/Video", 
            "", 
            "Image/Video Files (*.png *.jpg *.bmp *.mp4 *.avi)"
        )
        if self.file_path:
            self.process_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.display_file(self.file_path)
            self.status_label.setText("File loaded")
            # Clear previous violation log
            self.violation_log.setText("Violation Log:")

    def display_file(self, file_path):
        if file_path.lower().endswith(('.png', '.jpg', '.bmp')):
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
            )
        elif file_path.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(
                    rgb_image.data, 
                    w, h, 
                    bytes_per_line, 
                    QImage.Format_RGB888
                )
                p = convert_to_Qt_format.scaled(
                    self.image_label.width(), 
                    self.image_label.height(), 
                    Qt.KeepAspectRatio
                )
                self.image_label.setPixmap(QPixmap.fromImage(p))
            cap.release()

    def process_file(self):
        if self.file_path:
            self.process_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.status_label.setText("Processing...")
            self.yolo_thread.set_file(self.file_path)
            self.yolo_thread.start()

    def processing_finished(self):
        self.status_label.setText("Processing complete")
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def update_status(self, message):
        self.status_label.setText(message)

    def save_output(self):
        if not self.file_path:
            return

        # Get save file path from user
        file_dialog = QFileDialog()
        if self.file_path.lower().endswith(('.mp4', '.avi')):
            save_path, _ = file_dialog.getSaveFileName(
                self,
                "Save Video",
                "",
                "Video Files (*.mp4 *.avi)"
            )
            if save_path:
                self.save_video(save_path)
        else:
            save_path, _ = file_dialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "Image Files (*.png *.jpg *.bmp)"
            )
            if save_path:
                self.save_image(save_path)

    def save_video(self, save_path):
        if not self.yolo_thread.processed_frames:
            QMessageBox.warning(self, "Warning", "No processed video to save!")
            return

        self.status_label.setText("Saving video...")
        sample_frame = self.yolo_thread.processed_frames[0]
        height, width = sample_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
        
        for frame in self.yolo_thread.processed_frames:
            out.write(frame)
        
        out.release()
        self.status_label.setText("Video saved successfully!")
        QMessageBox.information(self, "Success", f"Video saved to:\n{save_path}")

    def save_image(self, save_path):
        if self.yolo_thread.current_frame is None:
            QMessageBox.warning(self, "Warning", "No processed image to save!")
            return

        self.status_label.setText("Saving image...")
        cv2.imwrite(save_path, self.yolo_thread.current_frame)
        self.status_label.setText("Image saved successfully!")
        QMessageBox.information(self, "Success", f"Image saved to:\n{save_path}")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, 
            w, h, 
            bytes_per_line, 
            QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        return QPixmap.fromImage(p)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())