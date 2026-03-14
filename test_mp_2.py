import mediapipe
try:
    import mediapipe.solutions
    print("Imported mediapipe.solutions")
    import mediapipe.python.solutions
    print("Imported mediapipe.python.solutions")
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
