import mediapipe as mp
try:
    print(mp.solutions)
    print("Success")
except AttributeError as e:
    print(f"Error: {e}")
    print(dir(mp))
