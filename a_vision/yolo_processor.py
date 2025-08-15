import os
import glob
from typing import List, Dict, Optional
from ultralytics import YOLO
import logging
from .database import PhotoDatabase
from .face_processor import FaceProcessor

logger = logging.getLogger(__name__)

class YOLOProcessor:
    def __init__(self, model_path: str = "yolov8n.pt", host: str = "localhost", 
                 user: str = "root", password: str = "", database: str = "photo_analysis", 
                 port: int = 3306, known_faces_folder: str = None, face_tolerance: float = 0.6):
        """
        Initialize YOLO processor with a model and database connection.
        
        Args:
            model_path: Path to YOLO model file (e.g., 'yolov8n.pt' or 'yolo11n_object365.pt')
            host: MySQL host address
            user: MySQL username
            password: MySQL password
            database: MySQL database name
            port: MySQL port number
            known_faces_folder: Path to folder containing known faces (optional)
            face_tolerance: Face recognition tolerance (0.0-1.0, lower = stricter)
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.db = PhotoDatabase(host, user, password, database, port)
        self.model_name = os.path.basename(model_path)
        
        # Initialize face processor
        self.face_processor = FaceProcessor(tolerance=face_tolerance)
        
        # Load known faces if folder is provided
        if known_faces_folder:
            self.load_known_faces(known_faces_folder)
        else:
            # Load known faces from database
            self.load_known_faces_from_database()
        
        logger.info(f"Initialized YOLO processor with model: {self.model_name}")
        logger.info(f"Face recognition enabled with {len(self.face_processor.known_face_names)} known faces")
    
    def process_single_image(self, image_path: str, confidence_threshold: float = 0.25) -> Dict:
        """
        Process a single image and store results in database.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Add photo to database (extracts EXIF data automatically)
            photo_id = self.db.add_photo(image_path, self.model_name)
            
            # Run YOLO detection
            results = self.model(image_path, conf=confidence_threshold)
            
            # Extract detection data
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = self.model.names[class_id]
                        
                        detections.append({
                            'class_name': class_name,
                            'class_id': class_id,
                            'confidence': confidence
                        })
            
            # Store detections in database
            if detections:
                self.db.add_object_detections(photo_id, detections)
            
            # Process face recognition
            face_detections = self.face_processor.detect_faces_in_image(image_path)
            if face_detections:
                self.db.add_face_detections(photo_id, face_detections)
            
            return {
                'photo_id': photo_id,
                'file_path': image_path,
                'detections_count': len(detections),
                'face_detections_count': len(face_detections),
                'detections': detections,
                'face_detections': face_detections
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def process_directory(self, directory_path: str, confidence_threshold: float = 0.25, 
                         file_extensions: List[str] = None, force_reprocess: bool = False) -> List[Dict]:
        """
        Process all images in a directory (including nested subdirectories) and store results in database.
        
        Args:
            directory_path: Path to directory containing images
            confidence_threshold: Minimum confidence for detections
            file_extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
            force_reprocess: If True, reprocess images even if already processed
            
        Returns:
            List of processing results for each image
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files recursively
        image_files = []
        for ext in file_extensions:
            # Search for both lowercase and uppercase extensions
            for case_ext in [ext, ext.upper()]:
                pattern = os.path.join(directory_path, f"**/*{case_ext}")
                found_files = glob.glob(pattern, recursive=True)
                image_files.extend(found_files)
        
        # Remove duplicates (in case both .jpg and .JPG patterns match the same file)
        image_files = list(set(image_files))
        image_files.sort()  # Sort for consistent processing order
        
        logger.info(f"Found {len(image_files)} images to process in {directory_path} (including subdirectories)")
        
        results = []
        processed_count = 0
        skipped_count = 0
        
        for image_path in image_files:
            try:
                # Check if image is already processed (unless force_reprocess is True)
                if not force_reprocess and self._is_image_already_processed(image_path):
                    logger.info(f"Skipped (already processed): {os.path.basename(image_path)}")
                    results.append({
                        'file_path': image_path,
                        'status': 'skipped',
                        'reason': 'already_processed',
                        'detections_count': 0
                    })
                    skipped_count += 1
                    continue
                
                result = self.process_single_image(image_path, confidence_threshold)
                result['status'] = 'processed'
                results.append(result)
                processed_count += 1
                logger.info(f"Processed: {os.path.basename(image_path)} - {result['detections_count']} objects")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'file_path': image_path,
                    'status': 'error',
                    'error': str(e),
                    'detections_count': 0
                })
        
        logger.info(f"Directory processing complete: {processed_count} processed, {skipped_count} skipped")
        return results
    
    def _is_image_already_processed(self, image_path: str) -> bool:
        """Check if an image has already been processed by looking for it in the database."""
        try:
            # Check if the photo exists in the database and has been processed
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT id, processed_at 
                FROM photos 
                WHERE file_path = %s
            ''', (image_path,))
            
            result = cursor.fetchone()
            if result:
                # Photo exists in database
                if result[1] is not None:  # processed_at is the second column
                    # Photo has been processed (has processed_at timestamp)
                    return True
                else:
                    # Photo exists but hasn't been processed yet
                    return False
            else:
                # Photo doesn't exist in database
                return False
                
        except Exception as e:
            logger.warning(f"Could not check if {image_path} is already processed: {e}")
            return False  # If we can't check, assume it needs processing
    
    def get_detection_summary(self, photo_id: int) -> Dict:
        """Get a summary of detections for a specific photo."""
        return self.db.get_photo_info(photo_id)
    
    def search_by_objects(self, class_names: List[str], min_count: int = 1) -> List[Dict]:
        """Search for photos containing specific objects."""
        return self.db.search_photos_by_objects(class_names, min_count)
    
    def get_database_statistics(self) -> Dict:
        """Get overall statistics from the database."""
        return self.db.get_statistics()
    
    def load_known_faces(self, known_faces_folder: str):
        """Load known faces from a folder and store them in the database."""
        logger.info(f"Loading known faces from: {known_faces_folder}")
        
        # Load faces from folder
        known_faces = self.face_processor.load_known_faces_from_folder(known_faces_folder)
        
        # Store in database
        for face_data in known_faces:
            try:
                self.db.add_known_face(
                    face_data['name'], 
                    face_data['face_encoding'], 
                    face_data['image_path']
                )
            except Exception as e:
                logger.error(f"Error storing known face {face_data['name']}: {e}")
        
        # Set faces for recognition
        self.face_processor.set_known_faces(known_faces)
        
        logger.info(f"Loaded and stored {len(known_faces)} known faces")
    
    def load_known_faces_from_database(self):
        """Load known faces from the database."""
        try:
            known_faces = self.db.get_known_faces()
            self.face_processor.set_known_faces(known_faces)
            logger.info(f"Loaded {len(known_faces)} known faces from database")
        except Exception as e:
            logger.warning(f"Could not load known faces from database: {e}")
    
    def search_by_faces(self, names: List[str]) -> List[Dict]:
        """Search for photos containing specific recognized faces."""
        return self.db.search_photos_by_faces(names)
    
    def get_face_statistics(self, photo_id: int) -> Dict:
        """Get face recognition statistics for a specific photo."""
        photo_info = self.db.get_photo_info(photo_id)
        if photo_info and 'face_summary' in photo_info:
            return photo_info['face_summary']
        return {}
    
    def close(self):
        """Close the database connection."""
        self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 