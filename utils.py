import cv2
from deepface import DeepFace
import time
import numpy as np
# import random
from queue import Empty, Full

# Function for video playback loop. Called by multiprocess

def video_loop(video_q, data_q, emotion_q):
    
    print("Video loops started")

    # define a video capture object
    vid = cv2.VideoCapture(0)

    ret, frame = vid.read()
    cv2.namedWindow('Emotional Window')
    cv2.imshow('Emotional Window', frame)

    emotion_label = "Emotions loading..."

    while(True):
        # tic = time.time()

        # Capture the video frame

        ret, frame = vid.read()

        # only pull emotion data from data queue when present. Max size of 1.
        try: 
            emotion_label = emotion_q.get(block = False)
        except:
            pass

        window_data = cv2.getWindowImageRect('Emotional Window') # returns 4-tuple with (x, y, width, height)
        # put emotions data to screen
        frame = cv2.putText(frame, emotion_label, (int(window_data[2] / 6), int(window_data[3] / 6)), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.imshow('Emotional Window', frame) # render

        # only put image frame into the video queue when empty. Max size of 1.
        try:
            video_q.put(frame, block = False)
        except:
            pass

        # toc = time.time()
        # print("frame rendering" + str(toc - tic))

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("stopping video loop")
            break

    # After quitting loop release capture object
    vid.release()
    video_q.put("STOP") # communicate to analysis loop
    # Destroy all the windows
    cv2.destroyAllWindows()

def analysis_loop(video_q, data_q, emotion_q):

    print("Emotion analysis loop started")

    while(True):

        # tic = time.time()

        # pull frame from queue
        # only pull from video queue when there is a frame present there
        try:
            analysis_frame = video_q.get(block = False)

            if analysis_frame == "STOP": # stop if STOP sent from camera
                print("stopping analysis loop")
                # data_q.get() # TODO: Change to clear queue
                data_q.put("STOP") # Send stop to music loop
                break

            data = DeepFace.analyze(analysis_frame, actions = ['emotion'], enforce_detection = False) # TODO: enforce detection, print something when no face

            # only put data when queue is empty
            try:
                data_q.put(data['dominant_emotion'], block = False)
            except:
                pass
            try:
                emotion_q.put(data['dominant_emotion'], block = False)
            except:
                pass

            # toc = time.time()
            # print("analysis loop" + str(toc - tic))
        except:
            pass

def music_loop(data_q, pause_time, emotion_params, genre_overlap, user_top_songs, user_top_artists, user_country, sp):

    print("Music generation loop started")

    previous_emotion = "Placeholder"

    while(True):

        # Pull emotion from queue
        # Only pull after emotions loaded. The try/except is to make sure queue is not empty
        # TODO: Fix loop ordering
        try:
            try:
                current_emotion = data_q.get(block = False)
            except:
                pass

            if current_emotion == "STOP": # Stop sent from data loop
                break

            if current_emotion != "Emotions loading..." and current_emotion != previous_emotion: # Makes sure data loop initialized and is not the same emotion

                previous_emotion = current_emotion
                current_emotion_params = emotion_params[current_emotion]
                # print(current_emotion_params)

                # print(current_emotion_params)
                print(current_emotion)

                # Get seeds for songs, artists, genres
                if np.random.rand() < 0.5:
                    song_seed = {np.random.choice(user_top_songs['items'])['uri']}
                else:
                    song_seed = []
                
                if np.random.rand() < 0.5:
                    artist_seed = {np.random.choice(user_top_artists['items'])['uri']}
                else:
                    artist_seed = []

                if np.random.rand() < 0.5 and np.size(genre_overlap) > 1:
                    genre_seeds = list(np.random.choice(genre_overlap, 2))
                else: 
                    genre_seeds = list(np.random.choice(genre_overlap, 1))

                # Get recommendations 
                recs = sp.recommendations(seed_artists = artist_seed, seed_genres = (genre_seeds), seed_tracks = song_seed, \
                    country = {user_country}, **current_emotion_params)
                # print(recs)
                print(np.size(recs['tracks']))

                # Choose song and play it
                song = np.random.choice(recs['tracks'])
                song_duration = song['duration_ms']
                offset = np.random.randint(song_duration // 6, song_duration // 3)
                sp.start_playback(uris = [song['uri']], position_ms = offset)

                print(sp.audio_features(song['uri']))

                time.sleep(pause_time)
                
        except Exception as e:
            print(e)
