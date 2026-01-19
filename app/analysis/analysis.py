from app.analysis.pose_estimation.pose_estimation import pose_estimation 

def analyze_video(filepath: str):
    print(f"Analyzing video: {filepath}")

    # pose processing
    estimation_2d, estimation_3d = pose_estimation(filepath)

    # other data / features Extraction

    # anaysis

    # report generation