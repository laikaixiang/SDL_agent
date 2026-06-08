import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "pos.json"


def _load_points() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as config:
        return json.load(config)


def parse_arm(pos_code:int) -> list:
    """
    Parses the move position configurations from pos.json for robot arm
    :param pos_code: The position code starting from 0
    :return: List of position parameters
    """
    points = _load_points()
    try:
        all_pos = points["RobotArm"]
        pos = all_pos[str(pos_code)]
        return pos
    except KeyError as e:
        print("No such position code")
        return []


def parse_dispenser(tip:int, action:int, pos_code:int) -> list:
    """
    Parses the move position configurations from pos.json for dispenser
    :param tip: The tip to use. 1 for left and 2 for right
    :param action: What action to take. 1 to get or throw tip and 2 to spit or suck liquid
    :param pos_code: The position code starting from 0
    :return: List of position parameters
    """
    points = _load_points()
    try:
        if tip == 1:
            dispenser_pos = points["LeftDispenser"]
            if action == 1:
                all_pos = dispenser_pos["Tip box"]
                pos = all_pos[str(pos_code)]
                return pos
            elif action == 2:
                all_pos = dispenser_pos["Bottle"]
                pos = all_pos[str(pos_code)]
                return pos
            else: raise ValueError("No such action code")
        elif tip == 2:
            dispenser_pos = points["RightDispenser"]
            if action == 1:
                all_pos = dispenser_pos["Tip box"]
                pos = all_pos[str(pos_code)]
                return pos
            elif action == 2:
                all_pos = dispenser_pos["Bottle"]
                pos = all_pos[str(pos_code)]
                return pos
            else: raise ValueError("No such action code")
        else: raise ValueError("No such tip")
    except KeyError as e:
        print("No such position code")
        return []
    except ValueError as e:
        print(str(e))
        return []
