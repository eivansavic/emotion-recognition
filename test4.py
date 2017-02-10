import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
facedict = {} # Kreiranje recnika za lica

def crop_face(gray, face):  # Odsecanje slike
    for (x, y, w, h) in face:
        faceslice = gray[y:y + h, x:x + w]

    facedict["face%s" %(len(facedict)+1)] = faceslice # Dodavanje isecene slike na kraj recnika
    return faceslice


while True:
    ret, frame = video_capture.read()  # Uzimanje frejma sa kamere. Povratna vrednost je 'true' ukoliko je frejm uspesno uhvacen.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Konvertovanje slike u crno-belu (nijanse sive) kako bi poboljsali brzinu prepoznavanja i tacnost.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # Pokretanje klasifikatora na frejm
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in face:  # Crtanje pravougaonika okolo detektovanog lica
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Crtanje pravougaonika pomocu OpenCV funkcije rectangle(coordinates, size, RGB color, thickness)

    if len(face) == 1:  # Jednostavna provera da li je lice detektovano, ukoliko nije ili ih ima vise na frejmu ispis "Nije pronadjeno ni jedno lice na kameri"
        faceslice = crop_face(gray, face)  # Odsecanje lica sa kamere
        cv2.imshow("detect", faceslice)  # Poseban prikaz odsecenog lica
    else:
        print("Nije pronadjeno ni jedno lice na kameri.")

    cv2.imshow("webcam", frame)  # Prikaz frejma

    if cv2.waitKey(1) & 0xFF == ord('q'):  # imshow ocekuje definisan prekid kako bi radio ispravno, postavljamo ga na 'q'
        break

    # Na kraju petlje postavljamo kriterijum za prekid izvrsavanja
    if len(facedict) == 10:
        break  # Program se zaustavlja kada se sakupi 10 slika lica