#!/usr/bin/env python3
"""
MySQL setup script for the photo analysis system.
This script helps you configure and test your MySQL database connection.
"""

import mysql.connector
import getpass
import sys

def test_connection(host, user, password, port=3306):
    """Test MySQL connection and return connection object if successful."""
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        print(f"✅ Successfully connected to MySQL server at {host}:{port}")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Failed to connect to MySQL: {err}")
        return None

def create_database(conn, database_name):
    """Create database if it doesn't exist."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        print(f"✅ Database '{database_name}' is ready")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Failed to create database: {err}")
        return False

def create_user(conn, username, password, database_name):
    """Create a new MySQL user with appropriate permissions."""
    try:
        cursor = conn.cursor()
        
        # Create user if it doesn't exist
        cursor.execute(f"CREATE USER IF NOT EXISTS '{username}'@'localhost' IDENTIFIED BY '{password}'")
        
        # Grant permissions
        cursor.execute(f"GRANT ALL PRIVILEGES ON {database_name}.* TO '{username}'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        print(f"✅ User '{username}' created with full permissions on database '{database_name}'")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Failed to create user: {err}")
        return False

def main():
    print("🔧 MySQL Setup for Photo Analysis System")
    print("=" * 50)
    
    # Get connection details
    print("\nEnter MySQL connection details:")
    host = input("Host (default: localhost): ").strip() or "localhost"
    port = input("Port (default: 3306): ").strip() or "3306"
    
    # Try to connect as root first
    print("\nAttempting to connect as root...")
    root_password = getpass.getpass("Root password (leave empty if none): ")
    
    conn = test_connection(host, "root", root_password, int(port))
    if not conn:
        print("\nTrying alternative connection methods...")
        
        # Try without password
        conn = test_connection(host, "root", "", int(port))
        if not conn:
            print("❌ Could not connect to MySQL server.")
            print("Please ensure MySQL is running and you have the correct credentials.")
            sys.exit(1)
    
    # Get database name
    database_name = input("\nDatabase name (default: photo_analysis): ").strip() or "photo_analysis"
    
    # Create database
    if not create_database(conn, database_name):
        sys.exit(1)
    
    # Ask if user wants to create a new user
    create_new_user = input("\nCreate a new user for this application? (y/n, default: y): ").strip().lower() or "y"
    
    if create_new_user == "y":
        username = input("New username (default: photo_user): ").strip() or "photo_user"
        password = getpass.getpass("New user password: ")
        
        if not create_user(conn, username, password, database_name):
            print("⚠️  User creation failed. You can still use root credentials.")
            username = "root"
            password = root_password
    else:
        username = "root"
        password = root_password
    
    # Test the final connection
    print(f"\nTesting connection with user '{username}'...")
    final_conn = test_connection(host, username, password, int(port))
    if not final_conn:
        print("❌ Final connection test failed.")
        sys.exit(1)
    
    # Test database access
    try:
        cursor = final_conn.cursor()
        cursor.execute(f"USE {database_name}")
        print(f"✅ Successfully connected to database '{database_name}'")
    except mysql.connector.Error as err:
        print(f"❌ Failed to access database: {err}")
        sys.exit(1)
    
    # Generate configuration
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nUse these credentials in your application:")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"User: {username}")
    print(f"Database: {database_name}")
    
    print("\nExample CLI usage:")
    print(f"python -m a_vision.cli process-image photo.jpg --host {host} --port {port} --user {username} --database {database_name}")
    
    print("\nExample Python usage:")
    print(f"from a_vision.yolo_processor import YOLOProcessor")
    print(f"processor = YOLOProcessor('yolov8n.pt', host='{host}', user='{username}', password='***', database='{database_name}', port={port})")
    
    # Close connections
    if conn:
        conn.close()
    if final_conn:
        final_conn.close()

if __name__ == "__main__":
    main() 