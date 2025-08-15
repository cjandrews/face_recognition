# Photo Analysis System

A comprehensive system for analyzing photos using YOLO object detection and EXIF data extraction. This system provides a normalized database schema to store and query photo metadata, object detection results, and camera information.

## Features

- **YOLO Object Detection**: Support for YOLOv8 and YOLO11 models
- **Face Recognition**: Detect and recognize faces using face_recognition library
- **EXIF Data Extraction**: Camera metadata, GPS coordinates, exposure settings
- **Normalized Database**: MySQL database with proper schema design
- **Command Line Interface**: Easy-to-use CLI for processing and querying
- **Batch Processing**: Process entire directories of images
- **Advanced Queries**: Search by objects, faces, analyze statistics, custom queries

## Database Schema

The system uses a normalized database design with four main tables:

### 1. `photos` table
- Basic photo information (file path, size, format)
- Processing metadata (model used, timestamps)

### 2. `exif_data` table
- Camera information (make, model, software)
- Exposure settings (shutter speed, aperture, ISO)
- GPS coordinates (latitude, longitude, altitude)
- Date/time information

### 3. `object_detections` table
- Individual object detections with confidence scores
- Links to photos and class information

### 4. `object_summary` table
- Aggregated object counts per photo
- Average and maximum confidence scores

### 5. `known_faces` table
- Face encodings of known people
- Name and source image information

### 6. `face_detections` table
- Individual face detections with locations
- Recognition results and confidence scores

### 7. `face_summary` table
- Aggregated face counts per photo
- Recognized vs unrecognized face statistics

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r a_vision/requirements.txt
   ```

2. **Set up MySQL Database**:
   - Install MySQL 8.0 or later
   - Create a database user with appropriate permissions
   - The system will automatically create the required tables

3. **Download YOLO Models** (optional - will download automatically):
   - YOLOv8n: `yolov8n.pt` (default)
   - YOLO11n Objects365: `yolo11n_object365.pt`

## Quick Start

### Command Line Interface

**Process a single image**:
```bash
python -m a_vision.cli process-image path/to/image.jpg --user your_user --password your_password
```

**Process all images in a directory**:
```bash
python -m a_vision.cli process-dir path/to/images/ --user your_user --password your_password
```

**Search for photos containing specific objects**:
```bash
python -m a_vision.cli search --objects person car --min-count 2 --user your_user --password your_password
```

**Get database statistics**:
```bash
python -m a_vision.cli stats --user your_user --password your_password
```

**View detailed photo information**:
```bash
python -m a_vision.cli info --photo-id 1 --user your_user --password your_password
```

**List all photos in database**:
```bash
python -m a_vision.cli list --user your_user --password your_password
```

**Load known faces from folder**:
```bash
python -m a_vision.cli load-faces path/to/known_faces/ --user your_user --password your_password
```

**Search for photos containing specific people**:
```bash
python -m a_vision.cli search-faces --names "John Doe" "Jane Smith" --user your_user --password your_password
```

**List all known faces in database**:
```bash
python -m a_vision.cli list-faces --user your_user --password your_password
```

### Face Recognition Setup

To use face recognition, you need to create a folder structure with known faces:

```
known_faces/
├── John_Doe/
│   ├── john1.jpg
│   ├── john2.jpg
│   └── john3.jpg
├── Jane_Smith/
│   ├── jane1.jpg
│   └── jane2.jpg
└── Other_Person/
    └── other1.jpg
```

Each subfolder should be named after the person, containing clear face images of that person.

### Programmatic Usage

```python
from a_vision.yolo_processor import YOLOProcessor

# Initialize processor with face recognition
with YOLOProcessor("yolov8n.pt", user="your_user", password="your_password", 
                  database="my_photos", known_faces_folder="known_faces") as processor:
    # Process a single image
    result = processor.process_single_image("photo.jpg")
    print(f"Detected {result['detections_count']} objects")
    print(f"Detected {result['face_detections_count']} faces")
    
    # Process a directory
    results = processor.process_directory("photos/")
    
    # Search for specific objects
    object_results = processor.search_by_objects(['person', 'car'])
    
    # Search for specific people
    face_results = processor.search_by_faces(['John_Doe', 'Jane_Smith'])
```
    photos_with_people = processor.search_by_objects(["person"], min_count=1)
    
    # Get statistics
    stats = processor.get_database_statistics()
```

## Advanced Usage

### Custom Confidence Thresholds

```bash
# Use higher confidence for more accurate detections
python -m a_vision.cli process-image photo.jpg --confidence 0.5
```

### Different YOLO Models

```bash
# Use Objects365 model for more object classes
python -m a_vision.cli process-image photo.jpg --model yolo11n_object365.pt
```

### Custom File Extensions

```bash
# Process specific file types
python -m a_vision.cli process-dir photos/ --extensions .jpg .png .tiff
```

### Custom Database Configuration

```bash
# Use a specific database
python -m a_vision.cli process-image photo.jpg --database my_custom_db --host remote-server.com --port 3307
```

## Example Queries

### Find Photos with High Object Counts

```sql
SELECT p.file_name, COUNT(os.id) as object_classes, SUM(os.total_count) as total_objects
FROM photos p
JOIN object_summary os ON p.id = os.photo_id
GROUP BY p.id
HAVING total_objects > 5
ORDER BY total_objects DESC;
```

### Find Photos with GPS Data

```sql
SELECT p.file_name, e.gps_latitude, e.gps_longitude
FROM photos p
JOIN exif_data e ON p.id = e.photo_id
WHERE e.gps_latitude IS NOT NULL AND e.gps_longitude IS NOT NULL;
```

### Find Photos by Camera Model

```sql
SELECT p.file_name, e.camera_make, e.camera_model
FROM photos p
JOIN exif_data e ON p.id = e.photo_id
WHERE e.camera_model LIKE '%iPhone%';
```

## API Reference

### PhotoDatabase Class

Main database interface for storing and querying photo data.

#### Methods:
- `add_photo(file_path, yolo_model=None)`: Add a photo to the database
- `add_object_detections(photo_id, detections)`: Store YOLO detection results
- `get_photo_info(photo_id)`: Get complete photo information
- `search_photos_by_objects(class_names, min_count=1)`: Search by objects
- `get_statistics()`: Get database statistics

### YOLOProcessor Class

High-level interface for processing images with YOLO models.

#### Methods:
- `process_single_image(image_path, confidence_threshold=0.25)`: Process one image
- `process_directory(directory_path, confidence_threshold=0.25)`: Process directory
- `get_detection_summary(photo_id)`: Get detection results
- `search_by_objects(class_names, min_count=1)`: Search functionality
- `get_database_statistics()`: Get statistics

## Configuration

### Environment Variables

- `YOLO_MODEL_PATH`: Default YOLO model path
- `MYSQL_HOST`: Default MySQL host (default: localhost)
- `MYSQL_USER`: Default MySQL username (default: root)
- `MYSQL_PASSWORD`: Default MySQL password
- `MYSQL_DATABASE`: Default MySQL database name (default: photo_analysis)
- `MYSQL_PORT`: Default MySQL port (default: 3306)
- `CONFIDENCE_THRESHOLD`: Default confidence threshold

### Model Options

- **YOLOv8n**: Fast, general-purpose object detection (80 classes)
- **YOLO11n Objects365**: More comprehensive object detection (365 classes)

## Performance Considerations

- **GPU Acceleration**: Install CUDA for faster processing
- **Batch Processing**: Process directories for better efficiency
- **Database Indexing**: MySQL tables include optimized indexes for common queries
- **Memory Usage**: Large images may require more RAM
- **MySQL Optimization**: Consider MySQL configuration tuning for large datasets

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Check internet connection, models download automatically
2. **Memory Errors**: Reduce batch size or use smaller images
3. **MySQL Connection Fails**: Check MySQL server status and credentials
4. **Database Permission Errors**: Ensure user has CREATE, INSERT, UPDATE, DELETE permissions
5. **EXIF Extraction Fails**: Some images may not have EXIF data

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the face_recognition repository and follows the same license terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example scripts
3. Open an issue on the repository 