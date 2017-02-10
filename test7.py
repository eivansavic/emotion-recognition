import cv2, numpy as np, argparse, time, glob, os, sys, subprocess, pandas, random, test3, math, xlrd

from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from termcolor import cprint
from pyfiglet import figlet_format

# Definisanje promenljivih i ucitavanje klasifikatora
camnumber = 0
video_capture = cv2.VideoCapture()
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fishface = cv2.createFisherFaceRecognizer()

try:
    fishface.load("trained_emoclassifier.xml")
except:
    print("Ne postoji utreniran xml fajl. Pokrenite program prvo sa parametrom --update")

parser = argparse.ArgumentParser(description="Options for the emotion-based music player")
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true")
args = parser.parse_args()

facedict = {}
actions = {}

emotions = ["neutral", "happy", "sadness"]
df = pandas.read_excel("EmotionLinks.xls")  # Fajl sa putanjama mp3 fajlova

actions["happy"] = [x for x in df.happy.dropna()]
actions["sadness"] = [x for x in df.sadness.dropna()]
actions["neutral"] = [x for x in df.neutral.dropna()]


def open_stuff(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


def crop_face(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y + h, x:x + w]
        faceslice = cv2.resize(faceslice, (350, 350))
    facedict["face%s" % (len(facedict) + 1)] = faceslice
    return faceslice


def update_model(emotions):
    print("Azuriranje modela aktivno")
    check_folders(emotions)
    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("Skupljanje slika...")
    print("Sledi azuriranje modela...")
    test3.update(emotions)
    print("Zavrseno!")


def check_folders(emotions):
    for x in emotions:
        if os.path.exists("dataset/%s" % x):
            pass
        else:
            os.makedirs("dataset/%s" % x)


def save_face(emotion):
    print("\n\nEmocija: " + emotion + ". Pritisni ENTER kada budes spreman da obucis skup...")
    raw_input()  # Stanje cekanja dok se ne pritisne ENTER
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(2)

    video_capture.open(camnumber)
    while len(facedict.keys()) < 16:
        detect_face()
    video_capture.release()
    for x in facedict.keys():
        cv2.imwrite("dataset/%s/%s.jpg" % (emotion, len(glob.glob("dataset/%s/*" % emotion))), facedict[x])
    facedict.clear()


def recognize_emotion():
    predictions = []
    confidence = []
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images/%s.jpg" % x, facedict[x])
        predictions.append(pred)
        confidence.append(conf)
    recognized_emotion = emotions[max(set(predictions), key=predictions.count)]
    print("Prepoznata emocija: %s" % recognized_emotion)
    actionlist = [x for x in actions[recognized_emotion]]  # Dobijanje liste akcija/fajlova za detekciju emocije
    random.shuffle(actionlist)  # Izmesaj listu
    open_stuff(actionlist[0])  # Otvori prvi u listi


def grab_webcamframe():
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # Primeni klasifikator na frejmu
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in face:  # Crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # OpenCv funkcija rectangle(coordinates, size, RGB color, thickness)

    cv2.imshow("webcam", frame)  # Prikaz frejma
    cv2.waitKey(1)

    return clahe_image


def detect_face():
    clahe_image = grab_webcamframe()
    time.sleep(2)
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face) == 1:
        faceslice = crop_face(clahe_image, face)
        print("Prepoznajem lice...")
        return faceslice
    else:
        print("Ne prepoznajem lice...")


def run_detection():
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(2)

    while len(facedict) != 2: # Koliko ce uzeti frame-ova sa kamere
        detect_face()
    recognize_emotion()


cprint(figlet_format('ivan', font='starwars'),
       'yellow', 'on_red', attrs=['bold'])
cprint(figlet_format('nikola', font='starwars'),
       'yellow', 'on_red', attrs=['bold'])
cprint(figlet_format('SOFT 2016/ 2017', font='starwars'),
       'yellow', 'on_red', attrs=['bold'])

print("Prepoznajem emociju...")
#pdate_model(emotions) #azuriranje dataseta :) :( :|

video_capture.open(camnumber)
run_detection()


