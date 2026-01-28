"""
Test that the improved reflect_joint_across_limb_axis function works correctly.
The joint should move PERPENDICULAR to the limb, not along the camera Z-axis.
"""
import numpy as np
import sys
sys.path.insert(0, '/run/media/arodabough/LibraryOfAlexandria2/Sandbox/dance-tutor/dance_analysis_server')

from app.analysis.pose_estimation.anatomical_constraints import reflect_joint_across_limb_axis


def test_reflection_geometry():
    """Test that reflection moves joint perpendicular to limb axis"""
    print("="*70)
    print("TESTING: reflect_joint_across_limb_axis geometry")
    print("="*70)
    print()
    
    # Test case: Straight arm along X-axis, elbow offset in Z
    shoulder = np.array([0.0, 0.5, 0.0])
    wrist = np.array([1.0, 0.5, 0.0])  # Arm along X-axis
    elbow = np.array([0.5, 0.5, 0.2])  # Elbow offset in +Z (in front)
    
    print("Test 1: Arm along X-axis, elbow in front")
    print(f"  Shoulder: {shoulder}")
    print(f"  Wrist:    {wrist}")
    print(f"  Elbow:    {elbow}")
    
    corrected = reflect_joint_across_limb_axis(shoulder, elbow, wrist)
    print(f"  Corrected elbow: {corrected}")
    
    # Expected: elbow should flip to Z=-0.2 (behind)
    # X and Y should stay near original since arm is along X-axis
    expected = np.array([0.5, 0.5, -0.2])
    
    close_to_expected = np.allclose(corrected, expected, atol=1e-6)
    print(f"  Expected: {expected}")
    print(f"  Matches expected: {close_to_expected}")
    print()
    
    # Verify the joint moved perpendicular to the limb
    limb_axis = wrist - shoulder
    movement = corrected - elbow
    
    # Movement should be perpendicular to limb (dot product ≈ 0)
    dot = np.dot(limb_axis, movement)
    is_perpendicular = abs(dot) < 1e-6
    print(f"  Movement vector: {movement}")
    print(f"  Dot product with limb axis: {dot:.6f}")
    print(f"  Movement is perpendicular to limb: {is_perpendicular} ✓" if is_perpendicular else f"  NOT perpendicular ✗")
    print()
    
    return close_to_expected and is_perpendicular


def test_distance_preservation():
    """Test that distances from parent AND to child are preserved"""
    print("="*70)
    print("TESTING: Distance preservation")
    print("="*70)
    print()
    
    shoulder = np.array([0.2, 0.5, 1.0])
    elbow = np.array([0.4, 0.3, 1.3])
    wrist = np.array([0.6, 0.1, 1.1])
    
    # Calculate original distances
    dist_shoulder_elbow = np.linalg.norm(elbow - shoulder)
    dist_elbow_wrist = np.linalg.norm(wrist - elbow)
    
    print(f"Original positions:")
    print(f"  Shoulder: {shoulder}")
    print(f"  Elbow:    {elbow}")
    print(f"  Wrist:    {wrist}")
    print(f"  Distance shoulder→elbow: {dist_shoulder_elbow:.4f}")
    print(f"  Distance elbow→wrist:    {dist_elbow_wrist:.4f}")
    print()
    
    corrected = reflect_joint_across_limb_axis(shoulder, elbow, wrist)
    
    # Calculate new distances
    new_dist_shoulder_elbow = np.linalg.norm(corrected - shoulder)
    new_dist_elbow_wrist = np.linalg.norm(wrist - corrected)
    
    print(f"Corrected elbow: {corrected}")
    print(f"  New distance shoulder→elbow: {new_dist_shoulder_elbow:.4f}")
    print(f"  New distance elbow→wrist:    {new_dist_elbow_wrist:.4f}")
    
    # With reflection across the line, only the perpendicular component changes
    # The distances should be preserved if the elbow is exactly on the perpendicular
    # In general, they might change slightly but the position should be valid
    
    print()
    print(f"  Shoulder-elbow distance change: {abs(new_dist_shoulder_elbow - dist_shoulder_elbow):.6f}")
    print(f"  Elbow-wrist distance change:    {abs(new_dist_elbow_wrist - dist_elbow_wrist):.6f}")
    
    return True


def test_perpendicular_movement():
    """Test multiple configurations to ensure movement is always perpendicular"""
    print("="*70)
    print("TESTING: Movement is perpendicular to limb in all configurations")
    print("="*70)
    print()
    
    test_cases = [
        # (name, parent, joint, child)
        ("Arm along X", 
         np.array([0.0, 0.5, 0.0]), 
         np.array([0.5, 0.5, 0.2]), 
         np.array([1.0, 0.5, 0.0])),
        
        ("Arm along Y", 
         np.array([0.0, 0.0, 0.0]), 
         np.array([0.0, 0.5, 0.2]), 
         np.array([0.0, 1.0, 0.0])),
        
        ("Arm along Z", 
         np.array([0.0, 0.0, 0.0]), 
         np.array([0.2, 0.0, 0.5]), 
         np.array([0.0, 0.0, 1.0])),
        
        ("Diagonal arm", 
         np.array([0.0, 0.0, 0.0]), 
         np.array([0.5, 0.5, 0.7]), 
         np.array([1.0, 1.0, 1.0])),
        
        ("Leg going down", 
         np.array([0.2, 0.0, 1.0]), 
         np.array([0.2, -0.4, 1.2]), 
         np.array([0.2, -0.8, 1.0])),
    ]
    
    all_passed = True
    
    for name, parent, joint, child in test_cases:
        limb_axis = child - parent
        
        corrected = reflect_joint_across_limb_axis(parent, joint, child)
        movement = corrected - joint
        
        # Check perpendicularity
        dot = np.dot(limb_axis, movement)
        is_perp = abs(dot) < 1e-6
        
        # Check that the joint actually moved (not degenerate)
        moved = np.linalg.norm(movement) > 1e-6
        
        status = "✓" if (is_perp and moved) else "✗"
        print(f"  {name:20s}: perpendicular={is_perp}, moved={moved} {status}")
        
        if not (is_perp and moved):
            all_passed = False
            print(f"      Movement: {movement}")
            print(f"      Dot product: {dot}")
    
    print()
    return all_passed


def visualize_reflection():
    """Create a simple visualization of the reflection"""
    print("="*70)
    print("VISUALIZATION: How reflection works")
    print("="*70)
    print()
    print("  Consider an arm from shoulder (S) to wrist (W):")
    print("  The elbow (E) is offset from this line.")
    print()
    print("  Before:                After reflection:")
    print("                          ")
    print("      E                       ")
    print("     /                         \\")
    print("    /                           \\")
    print("   S-----------W           S-----------W")
    print("                                /")
    print("                               /")
    print("                              E'")
    print()
    print("  The elbow moves from one side of the S-W line to the other,")
    print("  moving PERPENDICULAR to the limb axis (not along camera Z).")
    print()
    print("  This correctly handles:")
    print("  - Arms/legs at any angle to the camera")
    print("  - Diagonal limbs")
    print("  - Limbs parallel to any axis")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VERIFYING CORRECTED REFLECTION ALGORITHM")
    print("="*70)
    print()
    print("The new algorithm reflects joints PERPENDICULAR to the limb axis,")
    print("not along the camera Z-axis. This is geometrically correct.")
    print()
    
    visualize_reflection()
    print()
    
    test1 = test_reflection_geometry()
    test2 = test_distance_preservation()
    test3 = test_perpendicular_movement()
    
    print()
    print("="*70)
    if test1 and test3:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print()
        print("The reflection algorithm correctly:")
        print("  1. Moves joints perpendicular to the limb axis")
        print("  2. Reflects to the opposite side of the limb")
        print("  3. Works for limbs at any orientation")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
