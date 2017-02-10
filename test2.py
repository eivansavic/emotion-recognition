import cv2
import glob

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Definicija emocija


def detect_faces(emotion):
    files = glob.glob("sorted_set/%s/*" % emotion)  # Lista svih slika

    filenumber = 0
    for f in files:
        print
        "Izvrsava se..."

        frame = cv2.imread(f)  # Otvaranje slike
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konvertovanje slike u crno-belu (nijanse sive)

        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # Prolazimo kroz detektovana lica (face, face1, face2, face3) i ukoliko postoji lice zaustavljamo se
        if len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures == face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""   # Lice nije nadjeno na slici

        # Odseci i zapamti sliku
        for (x, y, w, h) in facefeatures:  # Uzimamo kordinate pravougaonika koji sadrzi lice
            print
            "Lice prepoznato u fajlu: %s" % f
            gray = gray[y:y + h, x:x + w]  # Promena velicine frejma

            try:
                out = cv2.resize(gray, (350, 350))  # Promena velicine kako bi svaka slika imala iste dimenzije (350x350)
                cv2.imwrite("dataset/%s/%s.jpg" % (emotion, filenumber), out)  # Sacuvaj sliku
            except:
                pass  # U slucaju greske prekini sa radom
        filenumber += 1  # Broj sacuvanih slika


for emotion in emotions:
    detect_faces(emotion)  # Poziv funkcije

