#!/usr/bin/env python3
"""
Example usage of the photo analysis system with YOLO object detection and EXIF data extraction.
"""

import os
from a_vision.yolo_processor import YOLOProcessor
from a_vision.database import PhotoDatabase

def example_single_image_processing():
    """Example: Process a single image and view results."""
    print("=== Single Image Processing Example ===")
    
    # Initialize processor with YOLOv8 model and face recognition
    with YOLOProcessor("yolov8n.pt", database="example_analysis", 
                      known_faces_folder="known_faces") as processor:
        # Process a single image (replace with your image path)
        image_path = "examples/obama.jpg"  # Using an example from the project
        
        if os.path.exists(image_path):
            result = processor.process_single_image(image_path, confidence_threshold=0.25)
            
            print(f"Processed: {result['file_path']}")
            print(f"Photo ID: {result['photo_id']}")
            print(f"Objects detected: {result['detections_count']}")
            print(f"Faces detected: {result['face_detections_count']}")
            
            # Show detailed info
            info = processor.get_detection_summary(result['photo_id'])
            if info:
                print(f"\nPhoto details:")
                print(f"  Size: {info['photo']['width']}x{info['photo']['height']}")
                print(f"  Format: {info['photo']['format']}")
                
                if info['exif']:
                    print(f"  Camera: {info['exif'].get('camera_make', 'Unknown')} {info['exif'].get('camera_model', '')}")
                
                if info['objects']:
                    print(f"  Objects found:")
                    for obj in info['objects']:
                        print(f"    - {obj['class_name']}: {obj['total_count']} (avg confidence: {obj['avg_confidence']:.2f})")
                
                # Show face recognition results
                if result['face_detections']:
                    print(f"  Faces found:")
                    for face in result['face_detections']:
                        if face.get('recognized_name'):
                            print(f"    - {face['recognized_name']} (confidence: {face['recognized_confidence']:.2f})")
                        else:
                            print(f"    - Unknown face (confidence: {face['confidence']:.2f})")
        else:
            print(f"Example image not found: {image_path}")
            print("Please provide a valid image path")

def example_directory_processing():
    """Example: Process all images in a directory."""
    print("\n=== Directory Processing Example ===")
    
    # Initialize processor with Objects365 model for more classes
    with YOLOProcessor("yolo11n_object365.pt", database="example_analysis") as processor:
        # Process all images in examples directory
        examples_dir = "examples"
        
        if os.path.exists(examples_dir):
            results = processor.process_directory(examples_dir, confidence_threshold=0.3)
            
            print(f"Processed {len(results)} images")
            
            # Show summary
            successful = sum(1 for r in results if 'error' not in r)
            total_objects = sum(r.get('detections_count', 0) for r in results if 'error' not in r)
            
            print(f"Successful: {successful}")
            print(f"Total objects detected: {total_objects}")
            
            # Show some interesting results
            for result in results[:3]:  # Show first 3 results
                if 'error' not in result:
                    print(f"\n{result['file_path']}: {result['detections_count']} objects")
                    if result['detections']:
                        # Group by class
                        class_counts = {}
                        for det in result['detections']:
                            class_counts[det['class_name']] = class_counts.get(det['class_name'], 0) + 1
                        
                        for class_name, count in class_counts.items():
                            print(f"  - {class_name}: {count}")
        else:
            print(f"Examples directory not found: {examples_dir}")

def example_search_and_analysis():
    """Example: Search for specific objects and analyze results."""
    print("\n=== Search and Analysis Example ===")
    
    with PhotoDatabase(database="example_analysis") as db:
        # Get database statistics
        stats = db.get_statistics()
        print(f"Database contains {stats['total_photos']} photos with {stats['total_objects_detected']} total objects")
        
        if stats['top_objects']:
            print("\nTop 5 most common objects:")
            for i, obj in enumerate(stats['top_objects'][:5], 1):
                print(f"  {i}. {obj['class_name']}: {obj['total']}")
        
        # Search for specific objects
        search_objects = ['person', 'car', 'dog', 'cat']
        for obj in search_objects:
            results = db.search_photos_by_objects([obj], min_count=1)
            if results:
                print(f"\nPhotos containing '{obj}': {len(results)}")
                # Show first 3 results
                for result in results[:3]:
                    print(f"  - {result['file_name']}: {result['total_count']} {obj}")
            else:
                print(f"\nNo photos containing '{obj}' found")

def example_face_recognition():
    """Example: Face recognition functionality."""
    print("\n=== Face Recognition Example ===")
    
    # Initialize processor with face recognition
    with YOLOProcessor("yolov8n.pt", database="example_analysis", 
                      known_faces_folder="known_faces") as processor:
        
        # Load known faces from folder
        print("Loading known faces...")
        processor.load_known_faces("known_faces")
        
        # Process an image with faces
        image_path = "examples/two_people.jpg"
        if os.path.exists(image_path):
            result = processor.process_single_image(image_path, confidence_threshold=0.25)
            
            print(f"\nProcessed: {result['file_path']}")
            print(f"Faces detected: {result['face_detections_count']}")
            
            if result['face_detections']:
                print("Face recognition results:")
                for i, face in enumerate(result['face_detections'], 1):
                    if face.get('recognized_name'):
                        print(f"  Face {i}: {face['recognized_name']} (confidence: {face['recognized_confidence']:.2f})")
                    else:
                        print(f"  Face {i}: Unknown (confidence: {face['confidence']:.2f})")
        
        # Search for photos containing specific people
        print("\nSearching for photos containing 'John':")
        results = processor.search_by_faces(['John'])
        if results:
            print(f"Found {len(results)} photos with John:")
            for result in results:
                print(f"  - {result['file_name']} (confidence: {result['recognized_confidence']:.2f})")
        else:
            print("No photos found with John")

def example_custom_queries():
    """Example: Custom database queries for analysis."""
    print("\n=== Custom Queries Example ===")
    
    with PhotoDatabase(database="example_analysis") as db:
        cursor = db.conn.cursor()
        
        # Find photos with high object counts
        cursor.execute('''
            SELECT p.file_name, COUNT(os.id) as object_classes, SUM(os.total_count) as total_objects
            FROM photos p
            JOIN object_summary os ON p.id = os.photo_id
            GROUP BY p.id
            HAVING total_objects > 5
            ORDER BY total_objects DESC
            LIMIT 5
        ''')
        
        high_object_photos = cursor.fetchall()
        if high_object_photos:
            print("Photos with most objects detected:")
            for photo in high_object_photos:
                print(f"  - {photo['file_name']}: {photo['total_objects']} objects in {photo['object_classes']} classes")
        
        # Find photos with GPS data
        cursor.execute('''
            SELECT p.file_name, e.gps_latitude, e.gps_longitude
            FROM photos p
            JOIN exif_data e ON p.id = e.photo_id
            WHERE e.gps_latitude IS NOT NULL AND e.gps_longitude IS NOT NULL
            LIMIT 5
        ''')
        
        gps_photos = cursor.fetchall()
        if gps_photos:
            print(f"\nPhotos with GPS data: {len(gps_photos)}")
            for photo in gps_photos:
                print(f"  - {photo['file_name']}: {photo['gps_latitude']:.6f}, {photo['gps_longitude']:.6f}")

def main():
    """Run all examples."""
    print("Photo Analysis System - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_single_image_processing()
        example_directory_processing()
        example_search_and_analysis()
        example_face_recognition()
        example_custom_queries()
        
        print("\n" + "=" * 50)
        print("Examples completed! Check the 'example_analysis' MySQL database for the data.")
        print("\nYou can also use the command-line interface:")
        print("  python -m a_vision.cli --help")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install ultralytics pillow mysql-connector-python face-recognition dlib")

if __name__ == "__main__":
    main() 