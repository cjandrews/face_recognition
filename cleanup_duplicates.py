#!/usr/bin/env python3
"""
Script to clean up duplicate face entries in the database.
"""

import mysql.connector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_duplicate_faces(host='localhost', user='root', password='admin', database='ai_dev_schema', port=3306):
    """Remove duplicate face entries, keeping only the most recent one for each image path."""
    
    # Connect to database
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Find duplicates based on image_path
        cursor.execute('''
            SELECT image_path, COUNT(*) as count
            FROM known_faces 
            WHERE image_path IS NOT NULL
            GROUP BY image_path 
            HAVING COUNT(*) > 1
        ''')
        
        duplicates = cursor.fetchall()
        
        if not duplicates:
            logger.info("No duplicate face entries found!")
            return
        
        logger.info(f"Found {len(duplicates)} image paths with duplicate entries")
        
        total_removed = 0
        
        for duplicate in duplicates:
            image_path = duplicate['image_path']
            count = duplicate['count']
            
            logger.info(f"Processing duplicates for: {image_path} ({count} entries)")
            
            # Get all entries for this image path, ordered by creation time (newest first)
            cursor.execute('''
                SELECT id, name, created_at
                FROM known_faces 
                WHERE image_path = %s
                ORDER BY created_at DESC
            ''', (image_path,))
            
            entries = cursor.fetchall()
            
            # Keep the first (most recent) entry, delete the rest
            for entry in entries[1:]:
                cursor.execute('DELETE FROM known_faces WHERE id = %s', (entry['id'],))
                logger.info(f"  Removed duplicate entry: {entry['name']} (ID: {entry['id']})")
                total_removed += 1
        
        conn.commit()
        logger.info(f"Cleanup completed! Removed {total_removed} duplicate entries.")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during cleanup: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    cleanup_duplicate_faces() 