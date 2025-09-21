import mysql.connector
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image, ExifTags
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhotoDatabase:
    def __init__(self, host: str = "localhost", user: str = "root", password: str = "", 
                 database: str = "photo_analysis", port: int = 3306):
        """Initialize the MySQL database connection and create tables if they don't exist."""
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        
        # Connect to MySQL server
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        
        # Create database if it doesn't exist
        cursor = self.conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cursor.execute(f"USE {database}")
        self.conn.commit()
        
        # Enable dict-like access to rows
        # Note: We'll use cursor(dictionary=True) when needed
        self.create_tables()
    
    def create_tables(self):
        """Create all necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Photos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INT AUTO_INCREMENT PRIMARY KEY,
                file_path VARCHAR(500) UNIQUE NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                file_size BIGINT,
                width INT,
                height INT,
                format VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP NULL,
                yolo_model_used VARCHAR(100),
                INDEX idx_file_path (file_path),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # EXIF data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exif_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                photo_id INT NOT NULL,
                camera_make VARCHAR(100),
                camera_model VARCHAR(100),
                date_time_original TIMESTAMP NULL,
                exposure_time VARCHAR(50),
                f_number DECIMAL(5,2),
                iso_speed INT,
                focal_length DECIMAL(8,2),
                gps_latitude DECIMAL(10,8),
                gps_longitude DECIMAL(11,8),
                gps_altitude DECIMAL(10,2),
                software VARCHAR(200),
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                INDEX idx_photo_id (photo_id),
                INDEX idx_camera_model (camera_model),
                INDEX idx_gps (gps_latitude, gps_longitude)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Object detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                photo_id INT NOT NULL,
                class_name VARCHAR(100) NOT NULL,
                class_id INT NOT NULL,
                confidence DECIMAL(5,4) NOT NULL,
                count INT DEFAULT 1,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                INDEX idx_photo_id (photo_id),
                INDEX idx_class_name (class_name),
                INDEX idx_confidence (confidence)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Object summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_summary (
                id INT AUTO_INCREMENT PRIMARY KEY,
                photo_id INT NOT NULL,
                class_name VARCHAR(100) NOT NULL,
                class_id INT NOT NULL,
                total_count INT NOT NULL,
                avg_confidence DECIMAL(5,4),
                max_confidence DECIMAL(5,4),
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                UNIQUE KEY unique_photo_class (photo_id, class_name),
                INDEX idx_photo_id (photo_id),
                INDEX idx_class_name (class_name),
                INDEX idx_total_count (total_count)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Known faces table (for storing face embeddings of known people)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                face_encoding BLOB NOT NULL,
                image_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_image_path (image_path),
                INDEX idx_name (name),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Face detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                photo_id INT NOT NULL,
                face_encoding BLOB NOT NULL,
                location_top INT NOT NULL,
                location_right INT NOT NULL,
                location_bottom INT NOT NULL,
                location_left INT NOT NULL,
                confidence DECIMAL(5,4),
                recognized_name VARCHAR(100),
                recognized_confidence DECIMAL(5,4),
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                INDEX idx_photo_id (photo_id),
                INDEX idx_recognized_name (recognized_name),
                INDEX idx_confidence (confidence)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Face summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_summary (
                id INT AUTO_INCREMENT PRIMARY KEY,
                photo_id INT NOT NULL,
                total_faces INT NOT NULL,
                recognized_faces INT NOT NULL,
                unrecognized_faces INT NOT NULL,
                FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                UNIQUE KEY unique_photo_faces (photo_id),
                INDEX idx_photo_id (photo_id),
                INDEX idx_total_faces (total_faces),
                INDEX idx_recognized_faces (recognized_faces)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
        
        # Add unique constraint to known_faces table if it doesn't exist
        self._ensure_known_faces_unique_constraint()
    
    def _ensure_known_faces_unique_constraint(self):
        """Ensure the unique constraint exists on known_faces.image_path."""
        cursor = self.conn.cursor()
        try:
            # Check if the unique constraint already exists
            cursor.execute('''
                SELECT COUNT(*) 
                FROM information_schema.table_constraints 
                WHERE table_schema = %s 
                AND table_name = 'known_faces' 
                AND constraint_name = 'unique_image_path'
            ''', (self.database,))
            
            constraint_exists = cursor.fetchone()[0] > 0
            
            if not constraint_exists:
                # Add the unique constraint
                cursor.execute('''
                    ALTER TABLE known_faces 
                    ADD UNIQUE KEY unique_image_path (image_path)
                ''')
                self.conn.commit()
                logger.info("Added unique constraint to known_faces.image_path")
            else:
                logger.info("Unique constraint already exists on known_faces.image_path")
                
        except Exception as e:
            logger.warning(f"Could not add unique constraint to known_faces table: {e}")
            # Don't raise the exception as this is not critical for basic functionality
    
    def extract_exif_data(self, image_path: str) -> Dict:
        """Extract EXIF data from an image file."""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if not exif:
                    return {}
                
                # Map EXIF tags to readable names
                exif_data = {}
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
                
                # Extract specific fields we're interested in
                result = {}
                
                # Camera info
                result['camera_make'] = exif_data.get('Make')
                result['camera_model'] = exif_data.get('Model')
                result['software'] = exif_data.get('Software')
                
                # Date/time
                date_time = exif_data.get('DateTimeOriginal')
                if date_time:
                    result['date_time_original'] = self._validate_and_parse_datetime(date_time)
                
                # Exposure settings
                result['exposure_time'] = self._convert_rational(exif_data.get('ExposureTime'))
                result['f_number'] = self._convert_rational(exif_data.get('FNumber'))
                result['iso_speed'] = exif_data.get('ISOSpeedRatings')
                result['focal_length'] = self._convert_rational(exif_data.get('FocalLength'))
                
                # GPS data
                if 'GPSInfo' in exif_data:
                    gps_info = exif_data['GPSInfo']
                    result['gps_latitude'] = self._convert_gps_to_decimal(
                        gps_info.get('GPSLatitude'), gps_info.get('GPSLatitudeRef')
                    )
                    result['gps_longitude'] = self._convert_gps_to_decimal(
                        gps_info.get('GPSLongitude'), gps_info.get('GPSLongitudeRef')
                    )
                    result['gps_altitude'] = gps_info.get('GPSAltitude')
                
                return result
                
        except Exception as e:
            logger.warning(f"Could not extract EXIF data from {image_path}: {e}")
            return {}
    
    def _convert_rational(self, value):
        """Convert IFDRational values to decimal floats."""
        if value is None:
            return None
        
        try:
            # Handle IFDRational objects (fractions)
            if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                if value.denominator == 0:
                    return None
                return float(value.numerator) / float(value.denominator)
            
            # Handle fraction strings like "26/10"
            if isinstance(value, str) and '/' in value:
                try:
                    num, denom = value.split('/')
                    return float(num) / float(denom)
                except (ValueError, ZeroDivisionError):
                    return None
            
            # Handle regular numbers
            return float(value)
        except Exception:
            return None
    
    def _convert_gps_to_decimal(self, gps_coords, ref):
        """Convert GPS coordinates from degrees/minutes/seconds to decimal."""
        if not gps_coords or not ref:
            return None
        
        try:
            degrees = float(gps_coords[0])
            minutes = float(gps_coords[1])
            seconds = float(gps_coords[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            if ref in ['S', 'W']:
                decimal = -decimal
                
            return decimal
        except (ValueError, IndexError):
            return None
    
    def add_photo(self, file_path: str, yolo_model: str = None) -> int:
        """Add a photo to the database and return its ID."""
        cursor = self.conn.cursor()
        
        try:
            # Get file info
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # Get image dimensions
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
            
            # Check if photo already exists
            cursor.execute('SELECT id FROM photos WHERE file_path = %s', (file_path,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                    UPDATE photos 
                    SET file_name = %s, file_size = %s, width = %s, height = %s, 
                        format = %s, yolo_model_used = %s
                    WHERE file_path = %s
                ''', (file_name, file_size, width, height, format_name, yolo_model, file_path))
                photo_id = existing[0]  # Use index instead of dict key
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO photos 
                    (file_path, file_name, file_size, width, height, format, yolo_model_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (file_path, file_name, file_size, width, height, format_name, yolo_model))
                photo_id = cursor.lastrowid
            
            # Extract and store EXIF data
            exif_data = self.extract_exif_data(file_path)
            if exif_data:
                # Delete existing EXIF data if updating
                cursor.execute('DELETE FROM exif_data WHERE photo_id = %s', (photo_id,))
                
                cursor.execute('''
                    INSERT INTO exif_data 
                    (photo_id, camera_make, camera_model, date_time_original, 
                     exposure_time, f_number, iso_speed, focal_length,
                     gps_latitude, gps_longitude, gps_altitude, software)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (photo_id, exif_data.get('camera_make'), exif_data.get('camera_model'),
                      exif_data.get('date_time_original'), exif_data.get('exposure_time'),
                      exif_data.get('f_number'), exif_data.get('iso_speed'),
                      exif_data.get('focal_length'), exif_data.get('gps_latitude'),
                      exif_data.get('gps_longitude'), exif_data.get('gps_altitude'),
                      exif_data.get('software')))
            
            self.conn.commit()
            logger.info(f"Added photo: {file_name} (ID: {photo_id})")
            return photo_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding photo {file_path}: {e}")
            raise
    
    def add_object_detections(self, photo_id: int, detections: List[Dict]):
        """Add YOLO object detection results for a photo."""
        cursor = self.conn.cursor()
        
        try:
            # Group detections by class
            class_counts = {}
            class_confidences = {}
            
            for detection in detections:
                class_name = detection['class_name']
                class_id = detection['class_id']
                confidence = detection['confidence']
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    class_confidences[class_name] = []
                
                class_counts[class_name] += 1
                class_confidences[class_name].append(confidence)
            
            # Clear existing detections for this photo
            cursor.execute('DELETE FROM object_detections WHERE photo_id = %s', (photo_id,))
            cursor.execute('DELETE FROM object_summary WHERE photo_id = %s', (photo_id,))
            
            # Insert individual detections
            for detection in detections:
                cursor.execute('''
                    INSERT INTO object_detections 
                    (photo_id, class_name, class_id, confidence)
                    VALUES (%s, %s, %s, %s)
                ''', (photo_id, detection['class_name'], detection['class_id'], 
                      detection['confidence']))
            
            # Insert summary data
            for class_name, count in class_counts.items():
                confidences = class_confidences[class_name]
                avg_confidence = sum(confidences) / len(confidences)
                max_confidence = max(confidences)
                class_id = next(d['class_id'] for d in detections if d['class_name'] == class_name)
                
                cursor.execute('''
                    INSERT INTO object_summary 
                    (photo_id, class_name, class_id, total_count, avg_confidence, max_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (photo_id, class_name, class_id, count, avg_confidence, max_confidence))
            
            # Update processed timestamp
            cursor.execute('''
                UPDATE photos SET processed_at = %s WHERE id = %s
            ''', (datetime.now(), photo_id))
            
            self.conn.commit()
            logger.info(f"Added {len(detections)} detections for photo ID {photo_id}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding detections for photo {photo_id}: {e}")
            raise
    
    def get_photo_info(self, photo_id: int) -> Dict:
        """Get complete information about a photo including EXIF and object detections."""
        cursor = self.conn.cursor(dictionary=True)
        
        # Get photo details
        cursor.execute('SELECT * FROM photos WHERE id = %s', (photo_id,))
        photo = cursor.fetchone()
        if not photo:
            return None
        
        # Get EXIF data
        cursor.execute('SELECT * FROM exif_data WHERE photo_id = %s', (photo_id,))
        exif = cursor.fetchone()
        
        # Get object summary
        cursor.execute('SELECT * FROM object_summary WHERE photo_id = %s', (photo_id,))
        objects = cursor.fetchall()
        
        return {
            'photo': photo,
            'exif': exif if exif else {},
            'objects': objects
        }
    
    def search_photos_by_objects(self, class_names: List[str], min_count: int = 1) -> List[Dict]:
        """Search for photos containing specific objects."""
        cursor = self.conn.cursor(dictionary=True)
        
        placeholders = ','.join(['%s' for _ in class_names])
        cursor.execute(f'''
            SELECT p.*, os.class_name, os.total_count, os.avg_confidence
            FROM photos p
            JOIN object_summary os ON p.id = os.photo_id
            WHERE os.class_name IN ({placeholders}) AND os.total_count >= %s
            ORDER BY p.created_at DESC
        ''', class_names + [min_count])
        
        return cursor.fetchall()
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor(dictionary=True)
        
        # Total photos
        cursor.execute('SELECT COUNT(*) as count FROM photos')
        total_photos = cursor.fetchone()['count']
        
        # Processed photos
        cursor.execute('SELECT COUNT(*) as count FROM photos WHERE processed_at IS NOT NULL')
        processed_photos = cursor.fetchone()['count']
        
        # Total objects detected
        cursor.execute('SELECT SUM(total_count) as total FROM object_summary')
        result = cursor.fetchone()
        total_objects = result['total'] if result['total'] else 0
        
        # Most common objects
        cursor.execute('''
            SELECT class_name, SUM(total_count) as total
            FROM object_summary
            GROUP BY class_name
            ORDER BY total DESC
            LIMIT 10
        ''')
        top_objects = cursor.fetchall()
        
        # Face statistics
        cursor.execute('SELECT COUNT(*) as count FROM known_faces')
        total_known_faces = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM face_detections')
        total_face_detections = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM face_detections WHERE recognized_name IS NOT NULL')
        total_recognized_faces = cursor.fetchone()['count']
        
        return {
            'total_photos': total_photos,
            'processed_photos': processed_photos,
            'total_objects_detected': total_objects,
            'top_objects': top_objects,
            'total_known_faces': total_known_faces,
            'total_face_detections': total_face_detections,
            'total_recognized_faces': total_recognized_faces
        }
    
    def add_known_face(self, name: str, face_encoding: bytes, image_path: str = None) -> int:
        """Add or update a known face encoding in the database."""
        cursor = self.conn.cursor()
        
        try:
            # Check if a face with this image path already exists
            cursor.execute('SELECT id FROM known_faces WHERE image_path = %s', (image_path,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                    UPDATE known_faces 
                    SET name = %s, face_encoding = %s
                    WHERE image_path = %s
                ''', (name, face_encoding, image_path))
                
                face_id = existing[0]  # Access by index, not by key
                self.conn.commit()
                logger.info(f"Updated known face: {name} (ID: {face_id})")
                return face_id
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO known_faces (name, face_encoding, image_path)
                    VALUES (%s, %s, %s)
                ''', (name, face_encoding, image_path))
                
                face_id = cursor.lastrowid
                self.conn.commit()
                logger.info(f"Added known face: {name} (ID: {face_id})")
                return face_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding/updating known face {name}: {e}")
            raise
    
    def get_known_faces(self) -> List[Dict]:
        """Get all known face encodings."""
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM known_faces ORDER BY name')
        return cursor.fetchall()
    
    def add_face_detections(self, photo_id: int, face_detections: List[Dict]):
        """Add face detection results for a photo."""
        cursor = self.conn.cursor()
        
        try:
            # Clear existing face detections for this photo
            cursor.execute('DELETE FROM face_detections WHERE photo_id = %s', (photo_id,))
            cursor.execute('DELETE FROM face_summary WHERE photo_id = %s', (photo_id,))
            
            total_faces = len(face_detections)
            recognized_faces = sum(1 for d in face_detections if d.get('recognized_name'))
            unrecognized_faces = total_faces - recognized_faces
            
            # Insert individual face detections
            for detection in face_detections:
                cursor.execute('''
                    INSERT INTO face_detections 
                    (photo_id, face_encoding, location_top, location_right, location_bottom, 
                     location_left, confidence, recognized_name, recognized_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (photo_id, detection['face_encoding'], detection['location_top'],
                      detection['location_right'], detection['location_bottom'],
                      detection['location_left'], detection.get('confidence'),
                      detection.get('recognized_name'), detection.get('recognized_confidence')))
            
            # Insert face summary
            cursor.execute('''
                INSERT INTO face_summary 
                (photo_id, total_faces, recognized_faces, unrecognized_faces)
                VALUES (%s, %s, %s, %s)
            ''', (photo_id, total_faces, recognized_faces, unrecognized_faces))
            
            self.conn.commit()
            logger.info(f"Added {total_faces} face detections for photo ID {photo_id}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding face detections for photo {photo_id}: {e}")
            raise
    
    def search_photos_by_faces(self, names: List[str]) -> List[Dict]:
        """Search for photos containing specific recognized faces."""
        cursor = self.conn.cursor(dictionary=True)
        
        placeholders = ','.join(['%s' for _ in names])
        cursor.execute(f'''
            SELECT DISTINCT p.*, fd.recognized_name, fd.recognized_confidence
            FROM photos p
            JOIN face_detections fd ON p.id = fd.photo_id
            WHERE fd.recognized_name IN ({placeholders})
            ORDER BY p.created_at DESC
        ''', names)
        
        return cursor.fetchall()
    
    def _validate_and_parse_datetime(self, date_time_str: str) -> str:
        """Validate and parse EXIF datetime string, returning None for invalid dates."""
        if not date_time_str or not isinstance(date_time_str, str):
            return None
        
        # Strip whitespace and check for empty or invalid values
        date_time_str = date_time_str.strip()
        if not date_time_str or date_time_str in ['0000:00:00 00:00:00', '0000:00:00', '']:
            logger.debug(f"Invalid EXIF datetime found: '{date_time_str}'")
            return None
        
        # Check for zeroed dates or times
        if '0000' in date_time_str or ':00:00:00' in date_time_str:
            parts = date_time_str.split(' ')
            if len(parts) >= 2:
                date_part = parts[0]
                time_part = parts[1]
                if date_part.startswith('0000') or time_part == '00:00:00':
                    logger.debug(f"Zeroed EXIF datetime found: '{date_time_str}'")
                    return None
        
        try:
            parsed_date = datetime.strptime(date_time_str, '%Y:%m:%d %H:%M:%S')
            # Check if date is in reasonable range
            min_date = datetime(1900, 1, 1)
            max_date = datetime(2100, 12, 31)
            if min_date <= parsed_date <= max_date:
                return parsed_date.isoformat()
            else:
                logger.debug(f"EXIF datetime out of reasonable range: '{date_time_str}'")
                return None
        except ValueError as e:
            logger.debug(f"Could not parse EXIF datetime '{date_time_str}': {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error parsing EXIF datetime '{date_time_str}': {e}")
            return None
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 