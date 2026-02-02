"""
Test script to send a video file to the analysis API endpoint and check processing status.

The analysis is asynchronous - videos are queued for processing and a job_id is returned.
Use the job_id to check status and retrieve results.
"""
import requests
import time
from pathlib import Path
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/v1/analyze"

def upload_video(video_path: str) -> dict:
    """
    Upload a video file to the analysis API endpoint
    
    Args:
        video_path: Path to the video file to upload
        
    Returns:
        Response JSON from the API containing job_id
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
            print(f"Upload successful! Job ID: {result.get('job_id')}")
            print(f"Status: {result.get('status')} (stage: {result.get('stage')})")
            return result
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {ANALYZE_ENDPOINT}")
            print("Make sure the server is running on localhost:8000")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Error uploading video: {e}")
            raise


def check_job_status(job_id: str) -> dict:
    """
    Check the status of a video analysis job
    
    Args:
        job_id: The job ID returned from upload_video
        
    Returns:
        Status response from the API
    """
    status_endpoint = f"{ANALYZE_ENDPOINT}/{job_id}/status"
    
    try:
        response = requests.get(status_endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking job status: {e}")
        raise


def get_job_results(job_id: str) -> dict:
    """
    Get the complete results of a finished job
    
    Args:
        job_id: The job ID returned from upload_video
        
    Returns:
        Results response from the API with download links
    """
    result_endpoint = f"{ANALYZE_ENDPOINT}/{job_id}/result"
    
    try:
        response = requests.get(result_endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving results: {e}")
        raise


def wait_for_completion(job_id: str, max_wait_seconds: int = 3600, poll_interval: int = 5) -> dict:
    """
    Wait for a job to complete and retrieve results
    
    Args:
        job_id: The job ID returned from upload_video
        max_wait_seconds: Maximum time to wait before timing out (default 1 hour)
        poll_interval: Time between status checks in seconds (default 5)
        
    Returns:
        Results response from the API
    """
    print(f"\nWaiting for job {job_id} to complete...")
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait_seconds:
            raise TimeoutError(f"Job {job_id} did not complete within {max_wait_seconds} seconds")
        
        try:
            status_response = check_job_status(job_id)
            current_status = status_response.get('status')
            
            print(f"  [{elapsed:.0f}s] Status: {current_status}", end='')
            
            if status_response.get('progress'):
                print(f" ({status_response['progress']}%)", end='')
            if status_response.get('message'):
                print(f" - {status_response['message']}", end='')
            
            print()
            
            if current_status == 'finished':
                print(f"\nJob completed in {elapsed:.1f} seconds")
                return get_job_results(job_id)
            elif current_status == 'failed':
                error = status_response.get('error', 'Unknown error')
                raise RuntimeError(f"Job failed: {error}")
            
            time.sleep(poll_interval)
            
        except (requests.exceptions.RequestException, RuntimeError):
            raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python upload_video.py <video_path> [--status <job_id>] [--results <job_id>] [--wait]")
        print()
        print("Examples:")
        print("  # Upload and get job ID:")
        print("  python upload_video.py path/to/video.mp4")
        print()
        print("  # Upload and wait for completion:")
        print("  python upload_video.py path/to/video.mp4 --wait")
        print()
        print("  # Check status of existing job:")
        print("  python upload_video.py --status <job_id>")
        print()
        print("  # Get results of completed job:")
        print("  python upload_video.py --results <job_id>")
        sys.exit(1)
    
    try:
        if sys.argv[1] == "--status" and len(sys.argv) > 2:
            # Check status of existing job
            job_id = sys.argv[2]
            status = check_job_status(job_id)
            print(json.dumps(status, indent=2))
            
        elif sys.argv[1] == "--results" and len(sys.argv) > 2:
            # Get results of existing job
            job_id = sys.argv[2]
            results = get_job_results(job_id)
            print(json.dumps(results, indent=2))
            
        else:
            # Upload video
            video_path = sys.argv[1]
            upload_response = upload_video(video_path)
            job_id = upload_response.get('job_id')
            
            # Check if --wait flag is provided
            if "--wait" in sys.argv:
                results = wait_for_completion(job_id)
                print("\nResults:")
                print(json.dumps(results, indent=2))
            else:
                print("\nTo check status:")
                print(f"  python upload_video.py --status {job_id}")
                print("\nTo get results when complete:")
                print(f"  python upload_video.py --results {job_id}")
                print("\nTo wait for completion:")
                print(f"  python upload_video.py {video_path} --wait")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
