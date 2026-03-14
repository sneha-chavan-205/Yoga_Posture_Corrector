def classify_pose(left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle):

    # T Pose
    if left_arm_angle > 160 and right_arm_angle > 160 and left_leg_angle > 160 and right_leg_angle > 160:
        return "T Pose"

    # Tree Pose
    elif left_leg_angle < 120 and right_leg_angle > 160:
        return "Tree Pose"

    # Warrior Pose
    elif left_leg_angle < 120 and right_leg_angle > 160 and left_arm_angle > 160 and right_arm_angle > 160:
        return "Warrior Pose"

    # Chair Pos
    #

    elif left_leg_angle < 120 and right_leg_angle < 120:
        return "Chair Pose"

    else:
        return "Unknown Pose"