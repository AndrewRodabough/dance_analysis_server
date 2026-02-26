"""Create report output from feature extraction results."""

import logging
from collections import Counter, defaultdict
from itertools import groupby
from typing import Dict, Any, List, Tuple, Optional

from ..feature_extraction.leg_straightening_timing import FAULT_DETAILS, DEFAULT_FAULT_DETAILS

logger = logging.getLogger(__name__)

TRANSITION_ONLY_FAULTS = {
    "WEIGHT_TRANSFER_WITH_MOVING_FOOT",
}


def generate_report(
    analysis_results: Dict[str, Any],
    total_frames: int,
    frame_rate: int = 60,
    person_id: int = 0,
) -> Dict[str, Any]:
    """
    Build a minimal report focused on leg straightening timing.
    """
    walk_data = analysis_results.get("2d_walks") or analysis_results.get("walks_straightening") or {}
    states = walk_data.get("states", [])
    faults = walk_data.get("faults", [])

    # Build both walk segments and state segments for different purposes
    walk_segments = _build_release_segments(states)
    state_segments = _build_state_segments(states)
    
    # Filter faults for reporting
    filtered_faults = _filter_faults_for_reporting(faults, state_segments)
    
    # Score walks based on filtered faults
    walk_scores = _score_walks(walk_segments, filtered_faults)
    overall_score = _overall_score_from_walks(walk_scores)
    overall_rating = _overall_rating(overall_score)
    
    # Score state segments for phase feedback
    state_scores = _score_segments(state_segments, filtered_faults)
    aggregate_scores = _aggregate_phase_scores(state_segments, state_scores)

    fault_counts = Counter(fault.get("type", "UNKNOWN") for fault in filtered_faults)
    top_fault_types = [fault_type for fault_type, _ in fault_counts.most_common(3)]

    summary, suggestions = _build_overview_summary(filtered_faults, fault_counts, overall_score, overall_rating)
    details = _build_overview_details(
        top_fault_types,
        fault_counts,
        person_id,
        overall_rating,
        aggregate_scores,
    )

    figures = _build_figures(walk_segments, filtered_faults, person_id)
    fault_entries = _build_fault_entries(filtered_faults, total_frames, person_id, state_segments, state_scores)

    report = {
        "Overview": {
            "Summary": summary,
            "Suggestions": suggestions,
            "Details": details,
            "Statistics": {
                "TODO": "",
            },
        },
        "Timestamps": {
            "Frames": total_frames,
            "Frame_Rate": frame_rate,
            "Figures": figures,
            "Faults": fault_entries,
        },
    }

    return report

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


def _build_overview_summary(
    faults: List[Dict[str, Any]],
    fault_counts: Counter,
    overall_score: float,
    overall_rating: str,
) -> Tuple[List[str], List[str]]:
    if not faults:
        summary = [
            "Leg straightening stayed consistent throughout the walk.",
            "Weight transfer appeared stable on each step.",
        ]
        suggestions = [
            "Maintain the current straightening timing and posture.",
        ]
        return summary, suggestions

    top_fault = fault_counts.most_common(1)[0][0] if fault_counts else "UNKNOWN"
    label = _fault_label(top_fault)
    summary = [
        f"Overall leg straightening score: {overall_rating} ({overall_score:.0f}%).",
        f"Most common issue: {label}.",
    ]

    suggestions = []
    for fault_type, _ in fault_counts.most_common(3):
        suggestion = _fault_detail(fault_type)["suggestion"]
        if suggestion not in suggestions:
            suggestions.append(suggestion)

    return summary, suggestions


def _build_overview_details(
    top_fault_types: List[str],
    fault_counts: Counter,
    person_id: int,
    overall_rating: str,
    aggregate_scores: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not top_fault_types:
        feedback = [
            {
                "Person_IDS": [person_id],
                "Description": "No leg-straightening issues detected.",
            }
        ]
    else:
        feedback = [
            {
                "Person_IDS": [person_id],
                "Description": _fault_detail(fault_type)["error"],
            }
            for fault_type in top_fault_types
        ]

    phase_feedback = _build_phase_feedback(aggregate_scores, person_id)
    feedback.extend(phase_feedback)

    return [
        {
            "Name": "Leg Straightening & Bending",
            "Rating": overall_rating,
            "Feedback": feedback,
        }
    ]


def _build_figures(
    walk_segments: List[Tuple[int, int]],
    faults: List[Dict[str, Any]],
    person_id: int,
) -> List[Dict[str, Any]]:
    if not walk_segments:
        return []

    figures = []
    for start, end in walk_segments:
        segment_faults = [fault for fault in faults if start <= fault.get("frame", -1) <= end]
        rating = _rating_from_fault_count(len(segment_faults))
        figures.append(
            {
                "Start": start,
                "End": end,
                "Name": "Forward Walk",
                "Rating": rating,
                "Person_IDS": [person_id],
            }
        )

    return figures


def _build_fault_entries(
    faults: List[Dict[str, Any]],
    total_frames: int,
    person_id: int,
    segments: List[Dict[str, Any]],
    segment_scores: Dict[int, str],
) -> List[Dict[str, Any]]:
    if not faults:
        return []

    grouped = _group_faults(faults)
    entries = []
    for start, end, fault in grouped:
        details = _fault_detail(fault.get("type", "UNKNOWN"), fault)
        phase_score = _phase_score_for_frame(start, segments, segment_scores)
        error_text = details["error"]
        if phase_score:
            error_text = f"{error_text} Phase score: {phase_score}."
        scrub_frame = min((start + end) // 2, max(total_frames - 1, 0))
        entries.append(
            {
                "Start": start,
                "End": end,
                "Scrub_Frame": scrub_frame,
                "Person_IDS": [person_id],
                "Severity": details["severity"],
                "Feedback": {
                    "Error": error_text,
                    "Suggestion": details["suggestion"],
                },
            }
        )

    return entries


def _build_release_segments(states: List[str]) -> List[Tuple[int, int]]:
    segments = []
    start = 0
    for idx in range(1, len(states)):
        if states[idx] == "RELEASE" and states[idx - 1] != "RELEASE":
            segments.append((start, idx - 1))
            start = idx
    segments.append((start, len(states) - 1))
    return segments


def _build_state_segments(states: List[str]) -> List[Dict[str, Any]]:
    segments = []
    current_idx = 0
    for state_name, group in groupby(states):
        length = len(list(group))
        segments.append({
            "state": state_name,
            "start": current_idx,
            "end": current_idx + length - 1,
        })
        current_idx += length
    return segments


def _score_segments(
    segments: List[Dict[str, Any]],
    faults: List[Dict[str, Any]],
) -> Dict[int, str]:
    segment_scores = {}
    for idx, segment in enumerate(segments):
        seg_faults = [
            fault for fault in faults
            if segment["start"] <= fault.get("frame", -1) <= segment["end"]
        ]
        error_values = [
            fault.get("error_deg")
            for fault in seg_faults
            if fault.get("error_deg") is not None
        ]
        if not error_values:
            segment_scores[idx] = "Correct"
            continue
        max_error = max(error_values)
        if max_error <= 10:
            segment_scores[idx] = "Close"
        else:
            segment_scores[idx] = "Incorrect"
    return segment_scores


def _aggregate_phase_scores(
    segments: List[Dict[str, Any]],
    segment_scores: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    aggregate: Dict[str, Dict[str, Any]] = {}
    for idx, segment in enumerate(segments):
        state = segment["state"]
        score = segment_scores.get(idx, "Correct")
        if state not in aggregate:
            aggregate[state] = {"correct": 0, "total": 0}
        aggregate[state]["total"] += 1
        if score == "Correct":
            aggregate[state]["correct"] += 1

    for state, counts in aggregate.items():
        total = counts["total"]
        correct = counts["correct"]
        counts["fraction"] = f"{correct}/{total}" if total else "0/0"
        counts["rate"] = (correct / total) if total else 0.0

    return aggregate


def _score_walks(walk_segments: List[Tuple[int, int]], faults: List[Dict[str, Any]]) -> Dict[int, str]:
    """Score each walk segment as Correct/Close/Incorrect based on faults."""
    walk_scores = {}
    for idx, (start, end) in enumerate(walk_segments):
        seg_faults = [fault for fault in faults if start <= fault.get("frame", -1) <= end]
        error_values = [
            fault.get("error_deg")
            for fault in seg_faults
            if fault.get("error_deg") is not None
        ]
        if not error_values:
            walk_scores[idx] = "Correct"
            continue
        max_error = max(error_values)
        if max_error <= 10:
            walk_scores[idx] = "Close"
        else:
            walk_scores[idx] = "Incorrect"
    return walk_scores


def _overall_score_from_walks(walk_scores: Dict[int, str]) -> float:
    """Calculate overall score as percentage of walk ratings."""
    if not walk_scores:
        return 100.0
    points = 0
    for score in walk_scores.values():
        if score == "Correct":
            points += 2
        elif score == "Close":
            points += 1
    total = len(walk_scores) * 2
    return (points / total) * 100 if total else 0.0


def _overall_score(segment_scores: Dict[int, str]) -> float:
    if not segment_scores:
        return 100.0
    points = 0
    for score in segment_scores.values():
        if score == "Correct":
            points += 2
        elif score == "Close":
            points += 1
    total = len(segment_scores) * 2
    return (points / total) * 100 if total else 0.0


def _overall_rating(score: float) -> str:
    if score >= 100:
        return "Perfect"
    if score >= 85:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 65:
        return "Fair"
    return "Needs Work"


def _build_phase_feedback(
    aggregate_scores: Dict[str, Dict[str, Any]],
    person_id: int,
) -> List[Dict[str, Any]]:
    suggestions = {
        "RELEASE": "Release phase needs cleaner standing-leg stability.",
        "PASSING": "Passing phase needs more consistent knee flexion.",
        "EXTENSION": "Extension phase needs smoother straightening timing.",
        "ARRIVAL": "Arrival phase needs stronger straight-leg finish.",
        "COMPLETED": "Completion phase needs steadier supporting leg control.",
    }

    feedback = []
    for state, stats in aggregate_scores.items():
        if stats.get("rate", 1.0) < 0.7:
            feedback.append(
                {
                    "Person_IDS": [person_id],
                    "Description": f"{state} phase: {stats['fraction']} correct. {suggestions.get(state, '')}",
                }
            )

    return feedback


def _phase_score_for_frame(
    frame: int,
    segments: List[Dict[str, Any]],
    segment_scores: Dict[int, str],
) -> Optional[str]:
    for idx, segment in enumerate(segments):
        if segment["start"] <= frame <= segment["end"]:
            return segment_scores.get(idx)
    return None


def _group_faults(faults: List[Dict[str, Any]], merge_gap: int = 2) -> List[Tuple[int, int, Dict[str, Any]]]:
    faults_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fault in faults:
        fault_type = fault.get("type", "UNKNOWN")
        faults_by_type[fault_type].append(fault)

    grouped = []
    for fault_type, items in faults_by_type.items():
        frames = sorted(fault.get("frame", 0) for fault in items)
        by_frame = {fault.get("frame", 0): fault for fault in items}
        if not frames:
            continue
        start = prev = frames[0]
        sample_fault = by_frame[start]
        for frame in frames[1:]:
            if frame <= prev + merge_gap:
                prev = frame
                continue
            grouped.append((start, prev, sample_fault))
            start = prev = frame
            sample_fault = by_frame[frame]
        grouped.append((start, prev, sample_fault))

    grouped.sort(key=lambda item: item[0])
    return grouped


def _filter_faults_for_reporting(
    faults: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not faults:
        return []

    grouped = _group_faults(faults, merge_gap=1)
    segment_lengths = {
        idx: segment["end"] - segment["start"] + 1
        for idx, segment in enumerate(segments)
    }

    kept_frames = set()
    for start, end, fault in grouped:
        fault_type = fault.get("type", "UNKNOWN")
        group_len = end - start + 1
        if group_len > 1:
            kept_frames.update(range(start, end + 1))
            continue

        segment_idx = _segment_index_for_frame(start, segments)
        if segment_idx is not None and segment_lengths.get(segment_idx, 0) <= 1:
            kept_frames.add(start)
            continue

        if fault_type in TRANSITION_ONLY_FAULTS:
            kept_frames.add(start)
            continue

    return [fault for fault in faults if fault.get("frame") in kept_frames]


def _segment_index_for_frame(frame: int, segments: List[Dict[str, Any]]) -> Optional[int]:
    for idx, segment in enumerate(segments):
        if segment["start"] <= frame <= segment["end"]:
            return idx
    return None


def _fault_detail(fault_type: str, fault: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    details = DEFAULT_FAULT_DETAILS.copy()
    details.update(FAULT_DETAILS.get(fault_type, {}))
    if fault:
        if fault.get("severity"):
            details["severity"] = fault["severity"]
        if fault.get("error"):
            details["error"] = fault["error"]
        if fault.get("suggestion"):
            details["suggestion"] = fault["suggestion"]
    return details


def _fault_label(fault_type: str) -> str:
    labels = {
        "SOFT_STANDING_LEG_IN_DRIVE": "soft standing leg in the drive",
        "NO_DRIVE_ACTION": "insufficient drive knee bend",
        "DROPPED_HEIGHT_IN_PASSING": "dropped height in passing",
        "STIFF_PASSING_LEG": "stiff passing leg",
        "EARLY_LOCK_STUMPING": "early leg lock",
        "SOFT_KNEE_ARRIVAL": "soft knee on arrival",
        "BUCKLED_STANDING_LEG": "buckled standing leg",
        "WEIGHT_TRANSFER_WITH_MOVING_FOOT": "weight transfer during foot travel",
    }
    return labels.get(fault_type, "leg timing issues")


def _rating_from_fault_count(fault_count: int) -> str:
    if fault_count == 0:
        return "Good"
    if fault_count <= 3:
        return "Fair"
    return "Poor"
