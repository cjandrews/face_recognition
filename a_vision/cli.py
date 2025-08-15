#!/usr/bin/env python3
"""
Command-line interface for photo analysis with YOLO object detection and EXIF data extraction.
"""

import argparse
import sys
import os
from pathlib import Path
from .yolo_processor import YOLOProcessor
from .database import PhotoDatabase
import json

def main():
    parser = argparse.ArgumentParser(
        description="Photo analysis with YOLO object detection and EXIF data extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python -m a_vision.cli process-image path/to/image.jpg

  # Process all images in a directory
  python -m a_vision.cli process-dir path/to/images/

  # Search for photos containing specific objects
  python -m a_vision.cli search --objects person car --min-count 2

  # Get database statistics
  python -m a_vision.cli stats

  # Get detailed info about a specific photo
  python -m a_vision.cli info --photo-id 1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process single image command
    process_image_parser = subparsers.add_parser('process-image', help='Process a single image')
    process_image_parser.add_argument('image_path', help='Path to the image file')
    process_image_parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    process_image_parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    process_image_parser.add_argument('--host', default='localhost', help='MySQL host')
    process_image_parser.add_argument('--user', default='root', help='MySQL username')
    process_image_parser.add_argument('--password', default='', help='MySQL password')
    process_image_parser.add_argument('--database', default='photo_analysis', help='MySQL database name')
    process_image_parser.add_argument('--port', type=int, default=3306, help='MySQL port')
    
    # Process directory command
    process_dir_parser = subparsers.add_parser('process-dir', help='Process all images in a directory')
    process_dir_parser.add_argument('directory_path', help='Path to directory containing images')
    process_dir_parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    process_dir_parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    process_dir_parser.add_argument('--force-reprocess', action='store_true', 
                                   help='Force reprocessing of images even if already processed')
    process_dir_parser.add_argument('--known-faces', help='Path to folder containing known faces for recognition')
    process_dir_parser.add_argument('--host', default='localhost', help='MySQL host')
    process_dir_parser.add_argument('--user', default='root', help='MySQL username')
    process_dir_parser.add_argument('--password', default='', help='MySQL password')
    process_dir_parser.add_argument('--database', default='photo_analysis', help='MySQL database name')
    process_dir_parser.add_argument('--port', type=int, default=3306, help='MySQL port')
    process_dir_parser.add_argument('--extensions', nargs='+', 
                                   default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                                   help='File extensions to process')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for photos by objects')
    search_parser.add_argument('--objects', nargs='+', required=True, help='Object classes to search for')
    search_parser.add_argument('--min-count', type=int, default=1, help='Minimum count of objects')
    search_parser.add_argument('--host', default='localhost', help='MySQL host')
    search_parser.add_argument('--user', default='root', help='MySQL username')
    search_parser.add_argument('--password', default='', help='MySQL password')
    search_parser.add_argument('--database', default='photo_analysis', help='MySQL database name')
    search_parser.add_argument('--port', type=int, default=3306, help='MySQL port')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Get database statistics')
    stats_parser.add_argument('--host', default='localhost', help='MySQL host')
    stats_parser.add_argument('--user', default='root', help='MySQL username')
    stats_parser.add_argument('--password', default='', help='MySQL password')
    stats_parser.add_argument('--database', default='photo_analysis', help='MySQL database name')
    stats_parser.add_argument('--port', type=int, default=3306, help='MySQL port')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get detailed info about a photo')
    info_parser.add_argument('--photo-id', type=int, required=True, help='Photo ID')
    info_parser.add_argument('--host', default='localhost', help='MySQL host')
    info_parser.add_argument('--user', default='root', help='MySQL username')
    info_parser.add_argument('--password', default='', help='MySQL password')
    info_parser.add_argument('--database', default='photo_analysis', help='MySQL database name')
    info_parser.add_argument('--port', type=int, default=3306, help='MySQL port')
    
    # List photos command
    list_parser = subparsers.add_parser('list', help='List all photos in database')
    list_parser.add_argument('--host', default='localhost', help='MySQL host')
    list_parser.add_argument('--user', default='root', help='MySQL username')
    list_parser.add_argument('--password', default='', help='MySQL password')
    list_parser.add_argument('--database', default='photo_analysis', help='MySQL database name')
    list_parser.add_argument('--port', type=int, default=3306, help='MySQL port')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of photos to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'process-image':
            process_single_image(args)
        elif args.command == 'process-dir':
            process_directory(args)
        elif args.command == 'search':
            search_photos(args)
        elif args.command == 'stats':
            show_statistics(args)
        elif args.command == 'info':
            show_photo_info(args)
        elif args.command == 'list':
            list_photos(args)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def process_single_image(args):
    """Process a single image and display results."""
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    with YOLOProcessor(args.model, args.host, args.user, args.password, args.database, args.port) as processor:
        result = processor.process_single_image(args.image_path, args.confidence)
        
        print(f"✅ Processed: {os.path.basename(args.image_path)}")
        print(f"   Photo ID: {result['photo_id']}")
        print(f"   Objects detected: {result['detections_count']}")
        
        if result['detections']:
            print("   Detections:")
            for detection in result['detections']:
                print(f"     - {detection['class_name']} (confidence: {detection['confidence']:.2f})")

def process_directory(args):
    """Process all images in a directory."""
    if not os.path.exists(args.directory_path):
        print(f"Error: Directory not found: {args.directory_path}")
        return
    
    with YOLOProcessor(args.model, args.host, args.user, args.password, args.database, args.port, 
                       known_faces_folder=args.known_faces) as processor:
        results = processor.process_directory(
            args.directory_path, 
            args.confidence, 
            args.extensions,
            args.force_reprocess
        )
        
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total images: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        if successful > 0:
            total_objects = sum(r.get('detections_count', 0) for r in results if 'error' not in r)
            print(f"   Total objects detected: {total_objects}")

def search_photos(args):
    """Search for photos containing specific objects."""
    with PhotoDatabase(args.host, args.user, args.password, args.database, args.port) as db:
        results = db.search_photos_by_objects(args.objects, args.min_count)
        
        if not results:
            print(f"No photos found containing {', '.join(args.objects)} (min count: {args.min_count})")
            return
        
        print(f"Found {len(results)} photos containing {', '.join(args.objects)}:")
        print()
        
        for result in results:
            print(f"📷 {result['file_name']}")
            print(f"   Path: {result['file_path']}")
            print(f"   {result['class_name']}: {result['total_count']} (avg confidence: {result['avg_confidence']:.2f})")
            print()

def show_statistics(args):
    """Display database statistics."""
    with PhotoDatabase(args.host, args.user, args.password, args.database, args.port) as db:
        stats = db.get_statistics()
        
        print("📊 Database Statistics:")
        print(f"   Total photos: {stats['total_photos']}")
        print(f"   Processed photos: {stats['processed_photos']}")
        print(f"   Total objects detected: {stats['total_objects_detected']}")
        
        if stats['top_objects']:
            print("\n🏆 Top 10 Objects:")
            for i, obj in enumerate(stats['top_objects'], 1):
                print(f"   {i:2d}. {obj['class_name']}: {obj['total']}")

def show_photo_info(args):
    """Show detailed information about a specific photo."""
    with PhotoDatabase(args.host, args.user, args.password, args.database, args.port) as db:
        info = db.get_photo_info(args.photo_id)
        
        if not info:
            print(f"Photo ID {args.photo_id} not found")
            return
        
        photo = info['photo']
        exif = info['exif']
        objects = info['objects']
        
        print(f"📷 Photo Details (ID: {args.photo_id})")
        print(f"   File: {photo['file_name']}")
        print(f"   Path: {photo['file_path']}")
        print(f"   Size: {photo['width']}x{photo['height']} ({photo['format']})")
        print(f"   File size: {photo['file_size']:,} bytes")
        print(f"   Model used: {photo['yolo_model_used']}")
        print(f"   Created: {photo['created_at']}")
        print(f"   Processed: {photo['processed_at']}")
        
        if exif:
            print("\n📸 EXIF Data:")
            if exif.get('camera_make') or exif.get('camera_model'):
                print(f"   Camera: {exif.get('camera_make', '')} {exif.get('camera_model', '')}")
            if exif.get('date_time_original'):
                print(f"   Date: {exif['date_time_original']}")
            if exif.get('exposure_time'):
                print(f"   Exposure: {exif['exposure_time']}")
            if exif.get('f_number'):
                print(f"   F-number: f/{exif['f_number']}")
            if exif.get('iso_speed'):
                print(f"   ISO: {exif['iso_speed']}")
            if exif.get('gps_latitude') and exif.get('gps_longitude'):
                print(f"   GPS: {exif['gps_latitude']:.6f}, {exif['gps_longitude']:.6f}")
        
        if objects:
            print(f"\n🎯 Objects Detected ({len(objects)} classes):")
            for obj in objects:
                print(f"   {obj['class_name']}: {obj['total_count']} (avg: {obj['avg_confidence']:.2f}, max: {obj['max_confidence']:.2f})")
        else:
            print("\n🎯 No objects detected")

def list_photos(args):
    """List all photos in the database."""
    with PhotoDatabase(args.host, args.user, args.password, args.database, args.port) as db:
        cursor = db.conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT p.id, p.file_name, p.width, p.height, p.processed_at,
                   COUNT(os.id) as object_classes
            FROM photos p
            LEFT JOIN object_summary os ON p.id = os.photo_id
            GROUP BY p.id
            ORDER BY p.created_at DESC
            LIMIT %s
        ''', (args.limit,))
        
        photos = cursor.fetchall()
        
        if not photos:
            print("No photos found in database")
            return
        
        print(f"📷 Recent Photos (showing {len(photos)} of {args.limit}):")
        print()
        
        for photo in photos:
            status = "✅" if photo['processed_at'] else "⏳"
            print(f"{status} ID {photo['id']:3d}: {photo['file_name']}")
            print(f"      Size: {photo['width']}x{photo['height']}")
            print(f"      Objects: {photo['object_classes']} classes")
            if photo['processed_at']:
                print(f"      Processed: {photo['processed_at']}")
            print()

if __name__ == '__main__':
    main() 