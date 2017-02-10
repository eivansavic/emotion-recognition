import cv2
import numpy as np
import argparse
import time
import glob
import os
import test3

video_capture = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fishface = cv2.createFisherFaceRecognizer()
try:
    fishface.load("trained_emoclassifier.xml")
except:
    print("no xml found. Using --update will create one.")
parser = argparse.ArgumentParser(description="Options for the emotion-based music player") # Kreiranje objekta za parsiranje
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true") # Dodavanje argumenta --update
args = parser.parse_args() # Cuvanje bilo kakvih argumenata u objekat

facedict = {}
emotions = ["neutral", "happy", "sadness"]


def crop_face(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
    facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice


def update_model(emotions):
    print("Azuriranje modela aktivno!")
    check_folders(emotions)
    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("Sakupljanje slika...")
    print("Azuriranje modela...")
    test3.update(emotions)
    print("Zavrseno!")


def check_folders(emotions): # Provera da li folder postoji, ukoliko ne posotji kreira se novi "/dataset"
    for x in emotions:
        if os.path.exists("dataset/%s" %x):
            pass
        else:
            os.makedirs("dataset/%s" %x)


def save_face(emotion):
    print("\n\nIzgledajte " + emotion + " dok vreme ne istekne i drzite izraz lica dok traju instrukcije.")
    for i in range(0,5): # Imate 5 sekundi dok ne procitate koji izraz je potrebno napraviti
        print(5-i)
        time.sleep(1)
    while len(facedict.keys()) < 16: # Uzima se 15 frejmova za svaku emociju
        detect_face()
    for x in facedict.keys(): #Cuvanje sadrzaja u fajl
        cv2.imwrite("dataset/%s/%s.jpg" %(emotion, len(glob.glob("dataset/%s/*" %emotion))), facedict[x])
    facedict.clear() # Brisanje recnika kako bi bio spreman za sledecu obradu i cuvanje


def recognize_emotion():
    predictions = []
    confidence = []
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images/%s.jpg" %x, facedict[x])
        predictions.append(pred)
        confidence.append(conf)
    print("Mislim da ste %s." %emotions[max(set(predictions), key=predictions.count)])


def grab_webcamframe():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    return clahe_image


def detect_face():
    clahe_image = grab_webcamframe()
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        faceslice = crop_face(clahe_image, face)
        return faceslice
    else:
        print("Nije pronadjeno ni jedno lice na kameri.")


while True:
    detect_face()
    if args.update: # Ako je prepoznat --update argument, poziva se funckija za azuriranje
        update_model(emotions)
        break
    elif len(facedict) == 10: # Ukoliko nije, redovno se pokrece program sa normalnim prepoznavanje izraza/emocija
        recognize_emotion()
        break