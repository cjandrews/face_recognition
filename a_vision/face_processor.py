import face_recognition
import numpy as np
import os
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self, tolerance: float = 0.6):
        """Initialize the face processor with recognition tolerance."""
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
    
    def load_known_faces_from_folder(self, folder_path: str) -> List[Dict]:
        """Load known faces from a folder structure where each subfolder is a person's name."""
        known_faces = []
        
        if not os.path.exists(folder_path):
            logger.error(f"Known faces folder does not exist: {folder_path}")
            return known_faces
        
        for person_name in os.listdir(folder_path):
            person_folder = os.path.join(folder_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue
            
            logger.info(f"Processing known faces for: {person_name}")
            
            for filename in os.listdir(person_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(person_folder, filename)
                    
                    try:
                        # Load and encode the face
                        face_encoding = self._encode_face_from_image(image_path)
                        if face_encoding is not None:
                            # Convert numpy array to bytes for database storage
                            face_encoding_bytes = pickle.dumps(face_encoding)
                            known_faces.append({
                                'name': person_name,
                                'face_encoding': face_encoding_bytes,
                                'image_path': image_path
                            })
                            logger.info(f"Loaded face for {person_name} from {filename}")
                        else:
                            logger.warning(f"No face found in {image_path}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
        
        logger.info(f"Loaded {len(known_faces)} known faces from {folder_path}")
        return known_faces
    
    def _encode_face_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face encoding from a single image."""
        try:
            # Load the image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return None
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                return None
            
            # Return the first face encoding (assuming one face per known image)
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Error encoding face from {image_path}: {e}")
            return None
    
    def detect_faces_in_image(self, image_path: str) -> List[Dict]:
        """Detect and recognize faces in an image."""
        try:
            # Load the image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.info(f"No faces detected in {image_path}")
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Recognize faces
            face_detections = []
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Check if this face matches any known faces
                recognized_name = None
                recognized_confidence = None
                
                if self.known_face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance=self.tolerance
                    )
                    
                    if True in matches:
                        # Get the best match
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, 
                            face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            recognized_name = self.known_face_names[best_match_index]
                            recognized_confidence = 1.0 - face_distances[best_match_index]
                
                # Convert face encoding to bytes for database storage
                face_encoding_bytes = pickle.dumps(face_encoding)
                
                face_detection = {
                    'face_encoding': face_encoding_bytes,
                    'location_top': face_location[0],
                    'location_right': face_location[1],
                    'location_bottom': face_location[2],
                    'location_left': face_location[3],
                    'confidence': 1.0,  # Face detection confidence
                    'recognized_name': recognized_name,
                    'recognized_confidence': recognized_confidence
                }
                
                face_detections.append(face_detection)
            
            logger.info(f"Detected {len(face_detections)} faces in {image_path}")
            return face_detections
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def set_known_faces(self, known_faces: List[Dict]):
        """Set the known faces for recognition."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for face_data in known_faces:
            if isinstance(face_data['face_encoding'], bytes):
                # If it's stored as bytes, unpickle it
                face_encoding = pickle.loads(face_data['face_encoding'])
            else:
                # If it's already a numpy array
                face_encoding = face_data['face_encoding']
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(face_data['name'])
        
        logger.info(f"Set {len(self.known_face_encodings)} known faces for recognition")
    
    def get_face_statistics(self, face_detections: List[Dict]) -> Dict:
        """Get statistics about face detections."""
        total_faces = len(face_detections)
        recognized_faces = sum(1 for d in face_detections if d.get('recognized_name'))
        unrecognized_faces = total_faces - recognized_faces
        
        # Get unique recognized names
        recognized_names = list(set(
            d['recognized_name'] for d in face_detections 
            if d.get('recognized_name')
        ))
        
        return {
            'total_faces': total_faces,
            'recognized_faces': recognized_faces,
            'unrecognized_faces': unrecognized_faces,
            'recognized_names': recognized_names
        } 