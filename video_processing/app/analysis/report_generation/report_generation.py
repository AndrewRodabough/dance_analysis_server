"""Create Report from feature extraction"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_report(phases: list[Any]) -> Dict[str, Any]:
    
    try:        
        log_analysis_summary()
        
        return {}
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise



from itertools import groupby
from collections import Counter

def log_analysis_summary(analysis_result: Dict[str, Any]):
    """
    Parses the analysis output to print a clean, human-readable timeline.
    Groups consecutive frames of the same state and summarizes faults within them.
    """
    states = analysis_result["states"]
    faults = analysis_result["faults"]
    
    print(f"\n{'='*20} CHA CHA WALK ANALYSIS TIMELINE {'='*20}")
    
    # 1. Group consecutive states into segments
    # Result: [(StateName, StartFrame, EndFrame), ...]
    segments = []
    current_idx = 0
    for state_name, group in groupby(states):
        length = len(list(group))
        segments.append({
            "state": state_name,
            "start": current_idx,
            "end": current_idx + length - 1
        })
        current_idx += length

    # 2. Map faults to these segments
    # We want to know which faults happened during which state segment
    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        state_name = segment["state"]
        
        # Filter faults that happened in this frame range
        segment_faults = [
            f for f in faults 
            if seg_start <= f["frame"] <= seg_end
        ]
        
        # Print the State Header
        print(f"\n[Frames {seg_start:03d} - {seg_end:03d}]  STATE: {state_name}")
        
        # 3. Summarize the Faults (Deduplicate)
        if not segment_faults:
            print("    ✔  Technique Clean")
        else:
            # Count how many frames triggered each fault type
            fault_counts = Counter(f["type"] for f in segment_faults)
            
            for fault_type, count in fault_counts.items():
                # Visual Indicator for severity
                severity = "WARNING" if count < 3 else "FAIL" 
                print(f"    ❌ {fault_type} (triggered in {count} frames)")
                
                # Context helper (Why it triggered)
                if fault_type == "SOFT_STANDING_LEG_IN_DRIVE":
                    print("       -> Standing leg bent before the push was finished.")
                elif fault_type == "STIFF_PASSING_LEG":
                    print("       -> Knee angle was too straight while crossing legs.")
                elif fault_type == "EARLY_LOCK_STUMPING":
                    print("       -> Leg locked straight while foot was still moving fast.")
                elif fault_type == "SOFT_KNEE_ARRIVAL":
                    print("       -> Weight transferred, but leg wasn't 180° straight.")

    print(f"\n{'='*60}\n")
