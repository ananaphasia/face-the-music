from deepface import DeepFace
import cv2
from datetime import datetime

time_start = datetime.now()

data = DeepFace.analyze(img_path = "2.jpg", actions = ["emotion"]) # possibilities: age, gender, emotion, race

print("--------------------------------")
print("\n", data)
print(datetime.now() - time_start)

# On my machine, takes about 1.5 seconds / frame to calculate emotions. 
# Plans: have both video input and model running, output emotions only when previous one has been outputted

# Video input: 

while True:
    # turn on camera
    # start reading emotions
    pass

# lol don't even need to use cv2, this can be done automatically