import subprocess

try:
    subprocess.run(['python', 'faceDetection.py'], check=True)
    subprocess.run(['python', 'faceRecognition.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
