"""Data structure to break down the portions of a ballroom figure"""

from enum import Enum as PyEnum

# Step needs where each foot is, what direction each foot is facing,
# where the body is facing, what part of the foot the weight is on for each foot,
# which part of the foot lands first, hip alignment, hip tilt, travel distance, weight transfer
# body sway. May need to be broken down smaller components if a change happens more quickly.


# Relative to Line of Dance
class Alignment(int, PyEnum):
    LOD = 0
    DIAGWALL = 45
    WALL = 90
    DIAGWALLALOD = 135
    ALOD = 180
    DIAGCENTERALOD = 225
    CENTER = 270
    DIAGCENTER = 315

# Range of angles with 0 being neutral, looking at the man's back. These are guesses as of Mar 16
# Angles are the angle of the man's upper spine relative to the floor
class Sway(tuple, PyEnum):
    NEUTRAL =   (0,0)
    LEFT =   (330, 0)
    SLEFT = (315,329) # Strong Left sway
    RIGHT =   (0, 30)
    SRIGHT = (31, 45) # Strong Right sway

# LSB = Left and slightly back. DLF = diagonal left and forward
class Placement(list, PyEnum):
    FORWARD = [0,2]
    BACK = [0,-2]
    LEFT = [-2,0]
    RIGHT = [2, 0]
    LSF = [-2, 1]
    LSB = [-2,-1]
    RSF = [2, 1]
    RSB = [2,-1]
    FSL = [-1, 2]
    FSR = [1, 2]
    BSL = [-1, -2]
    BSR = [1, -2]
    DLF = [-2, 2]
    DLB = [-2, -2]
    DRF = [2, 2]
    DRB = [2, -2]
    NEUTRAL = [0, 0]


class Foot():
    placement: Placement
    alignment: Alignment

class Step():
    leftFoot: Foot
    rightFoot: Foot
    beats: int #How many counts this step should take. .5-2 usually

class Figure():
    name: str
    steps: list[Step]
