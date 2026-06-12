"""Vision processes: MediaPipe hand tracking (teleop) and YOLO/grasp (autonomous).

Heavy deps (mediapipe, ultralytics, opencv) are imported lazily inside the
process entry points so that the pure-logic modules here (e.g. gesture
debouncing) import and test without them.
"""
