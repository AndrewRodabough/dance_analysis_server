"""
Test script to send a video file to the analysis API endpoint and check processing status.

The analysis is asynchronous - videos are queued for processing and a job_id is returned.
Use the job_id to check status and retrieve results.

This script supports two upload methods:
1. Direct upload (RECOMMENDED): Video uploaded directly to S3 using presigned URL
2. Legacy upload: Video uploaded through API server (deprecated)
"""
import requests
import time
from pathlib import Path
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/v1/analyze"


def build_auth_headers(token: str = None) -> dict:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def upload_video_direct(video_path: str, token: str = None) -> dict:
    """
    Upload a video file directly to S3 using presigned URL (RECOMMENDED)
    
    This is more efficient as the video doesn't go through the API server.
    
    Args:
        video_path: Path to the video file to upload
        
    Returns:
        Response JSON from the API containing job_id
    """
    video_file = Path(video_path)
    
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Uploading video directly to S3: {video_path}")
    
    # Step 1: Request presigned upload URL
    print("  Step 1/3: Requesting presigned upload URL...")
    try:
        response = requests.post(
            f"{ANALYZE_ENDPOINT}/upload-url",
            params={
                "filename": video_file.name,
                "content_type": "video/mp4"
            },
            headers=build_auth_headers(token)
        )
        response.raise_for_status()
        url_data = response.json()
        
        job_id = url_data['job_id']
        upload_url = url_data['upload_url']
        s3_key = url_data['s3_key']
        
        print(f"  ✓ Received upload URL for job: {job_id}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error requesting upload URL: {e}")
        raise
    
    # Step 2: Upload video directly to S3
    print("  Step 2/3: Uploading video to S3...")
    try:
        with open(video_file, "rb") as f:
            video_content = f.read()
            
        response = requests.put(
            upload_url,
            data=video_content,
            headers={'Content-Type': 'video/mp4'}
        )
        response.raise_for_status()
        
        print(f"  ✓ Video uploaded to S3 ({len(video_content)} bytes)")
        
    except requests.exceptions.RequestException as e:
        print(f"Error uploading to S3: {e}")
        raise
    
    # Step 3: Confirm upload and start processing
    print("  Step 3/3: Confirming upload and starting analysis...")
    try:
        response = requests.post(
            f"{ANALYZE_ENDPOINT}/confirm",
            params={
                "job_id": job_id,
                "s3_key": s3_key
            },
            headers=build_auth_headers(token)
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Upload successful! Job ID: {result.get('job_id')}")
        print(f"  Status: {result.get('status')} (stage: {result.get('stage')})")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error confirming upload: {e}")
        raise


def upload_video(video_path: str, token: str = None) -> dict:
    """
    Upload a video file to the analysis API endpoint (LEGACY)
    
    This method uploads through the API server. For better performance,
    use upload_video_direct() instead.
    
    Args:
        video_path: Path to the video file to upload
        
    Returns:
        Response JSON from the API containing job_id
    """
    video_file = Path(video_path)
    
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Uploading video (legacy method): {video_path}")
    print("  Note: Consider using --direct flag for better performance")
    
    with open(video_file, "rb") as f:
        files = {"file": (video_file.name, f, "video/mp4")}
        
        try:
            response = requests.post(
                ANALYZE_ENDPOINT,
                files=files,
                headers=build_auth_headers(token)
            )
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


def check_job_status(job_id: str, token: str = None) -> dict:
    """
    Check the status of a video analysis job
    
    Args:
        job_id: The job ID returned from upload_video
        
    Returns:
        Status response from the API
    """
    status_endpoint = f"{ANALYZE_ENDPOINT}/{job_id}/status"
    
    try:
        response = requests.get(status_endpoint, headers=build_auth_headers(token))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking job status: {e}")
        raise


def get_job_results(job_id: str, token: str = None) -> dict:
    """
    Get the complete results of a finished job
    
    Args:
        job_id: The job ID returned from upload_video
        
    Returns:
        Results response from the API with download links
    """
    result_endpoint = f"{ANALYZE_ENDPOINT}/{job_id}/result"
    
    try:
        response = requests.get(result_endpoint, headers=build_auth_headers(token))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving results: {e}")
        raise


def wait_for_completion(
    job_id: str,
    token: str = None,
    max_wait_seconds: int = 3600,
    poll_interval: int = 5
) -> dict:
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
            status_response = check_job_status(job_id, token)
            current_status = status_response.get('status')
            
            print(f"  [{elapsed:.0f}s] Status: {current_status}", end='')
            
            if status_response.get('progress'):
                print(f" ({status_response['progress']}%)", end='')
            if status_response.get('message'):
                print(f" - {status_response['message']}", end='')
            
            print()
            
            if current_status == 'finished':
                print(f"\nJob completed in {elapsed:.1f} seconds")
                return get_job_results(job_id, token)
            elif current_status == 'failed':
                error = status_response.get('error', 'Unknown error')
                raise RuntimeError(f"Job failed: {error}")
            
            time.sleep(poll_interval)
            
        except (requests.exceptions.RequestException, RuntimeError):
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload a video for analysis and monitor job status."
    )
    parser.add_argument("video_path", nargs="?", help="Path to the video file")
    parser.add_argument("--direct", action="store_true", help="Use direct S3 upload")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument("--status", help="Check status of an existing job_id")
    parser.add_argument("--results", help="Fetch results of an existing job_id")
    parser.add_argument("--token", help="Bearer token for authenticated requests")

    args = parser.parse_args()

    try:
        if args.status:
            status = check_job_status(args.status, args.token)
            print(json.dumps(status, indent=2))
            raise SystemExit(0)

        if args.results:
            results = get_job_results(args.results, args.token)
            print(json.dumps(results, indent=2))
            raise SystemExit(0)

        if not args.video_path:
            parser.print_help()
            raise SystemExit(1)

        if args.direct:
            upload_response = upload_video_direct(args.video_path, args.token)
        else:
            upload_response = upload_video(args.video_path, args.token)

        job_id = upload_response.get("job_id")

        if args.wait:
            results = wait_for_completion(job_id, args.token)
            print("\nResults:")
            print(json.dumps(results, indent=2))
        else:
            print("\nTo check status:")
            print(f"  python upload_video.py --status {job_id} --token <token>")
            print("\nTo get results when complete:")
            print(f"  python upload_video.py --results {job_id} --token <token>")
            print("\nTo wait for completion:")
            print(f"  python upload_video.py {args.video_path} --direct --wait --token <token>")

    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)
