import os
import sys
import mediapipe

print(f"Python version: {sys.version}")
mp_path = os.path.dirname(mediapipe.__file__)
print(f"Mediapipe path: {mp_path}")

try:
    print(f"Contents of mediapipe dir: {os.listdir(mp_path)}")
except Exception as e:
    print(f"Error listing dir: {e}")

sol_path = os.path.join(mp_path, 'python', 'solutions')
print(f"Checking for solutions at: {sol_path}")
print(f"Exists: {os.path.exists(sol_path)}")

if os.path.exists(sol_path):
    print(f"Contents of solutions: {os.listdir(sol_path)}")

# Check for task api
tasks_path = os.path.join(mp_path, 'tasks')
print(f"Tasks contents: {os.listdir(tasks_path) if os.path.exists(tasks_path) else 'Missing'}")
