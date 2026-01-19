"""
Test script to send a video file to the analysis API endpoint
"""
import requests
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{API_BASE_URL}/analyze"

def upload_video(video_path: str) -> dict:
    """
    Upload a video file to the analysis API endpoint
    
    Args:
        video_path: Path to the video file to upload
        
    Returns:
        Response JSON from the API
    """
    video_file = Path(video_path)
    
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Uploading video: {video_path}")
    
    with open(video_file, "rb") as f:
        files = {"file": (video_file.name, f, "video/mp4")}
        
        try:
            response = requests.post(ANALYZE_ENDPOINT, files=files)
            response.raise_for_status()
            result = response.json()
            print(f"Success! Response: {result}")
            return result
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {ANALYZE_ENDPOINT}")
            print("Make sure the server is running on localhost:8000")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Error uploading video: {e}")
            raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        exit("Please provide the path to the video file as a command line argument.")
    
    try:
        upload_video(video_path)
    except Exception as e:
        print(f"Failed to upload video: {e}")
        sys.exit(1)
