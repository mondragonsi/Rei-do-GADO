import mediapipe as mp
try:
    from mediapipe.tasks.python import vision
    print("Success: Imported mediapipe.tasks.python.vision")
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
