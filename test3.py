import cv2
import glob
import random
import numpy as np

emotions = ["neutral", "happy", "sadness"]  # Lista emocija

#help(cv2)

fishface = cv2.createFisherFaceRecognizer()  # Inicijalizacije Fisher Face klasifikatora

data = {}


# def get_files(emotion):  # Funkcija koja uzima listu fajlova, na slucajan nacin ih izmesa i podeli na dva dela 80:20
#     files = glob.glob("dataset/%s/*" % emotion)
#     random.shuffle(files)
#     training = files[:int(len(files) * 0.8)]  # Prvih 80% slika koristimo za obucavanje
#     prediction = files[-int(len(files) * 0.2):]  # Ostalih 20% slika za predikciju
#     return training, prediction
#
#
# def make_sets():
#     training_data = []
#     training_labels = []
#     prediction_data = []
#     prediction_labels = []
#     for emotion in emotions:
#         training, prediction = get_files(emotion)
#         # Dodavanje podataka na listu za obucavanje i predikcije i generisanje labela 0-2 (broj emocija)
#         for item in training:
#             image = cv2.imread(item)  # Otvaranje slike
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konvertovanje slike u crno-belu (nijanse sive)
#             training_data.append(gray)  # Dodavanje konvertovanih slika na trening listu
#             training_labels.append(emotions.index(emotion))
#
#         for item in prediction:  # Ponavljanje procesa za predikciju
#             image = cv2.imread(item)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             prediction_data.append(gray)
#             prediction_labels.append(emotions.index(emotion))
#
#     return training_data, training_labels, prediction_data, prediction_labels
#
#
# def run_recognizer():
#     training_data, training_labels, prediction_data, prediction_labels = make_sets()
#
#     print ("Trenira se Fisher Face klasifikator...")
#     print ("Broj slika u trening skupu je: ", len(training_labels))
#     fishface.train(training_data, np.asarray(training_labels))
#
#     print ("Predikcija klasifikacionog seta...")
#     cnt = 0
#     correct = 0
#     incorrect = 0
#     for image in prediction_data:
#         pred, conf = fishface.predict(image)
#         if pred == prediction_labels[cnt]:
#             correct += 1
#             cnt += 1
#         else:
#             incorrect += 1
#             cnt += 1
#     return ((100 * correct) / (correct + incorrect))


# # Pokretanje
# metascore = []
# for i in range(0, 10):
#     correct = run_recognizer()
#     print ("Tacnost: ", correct, "%")
#     metascore.append(correct)
#
# print ("\n\n Zavrseno! Ukupna tacnost: ", np.mean(metascore), "%")

def make_sets(emotions):
    training_data = []
    training_labels = []

    for emotion in emotions:
        training = glob.glob("dataset/%s/*" % emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

    return training_data, training_labels


def run_recognizer(emotions):
    training_data, training_labels = make_sets(emotions)

    print("Treniranje Fisher Face klasifikatora")
    print("Broj slika za trening set je: " + str(len(training_labels)))
    fishface.train(training_data, np.asarray(training_labels))


def update(emotions):
    run_recognizer(emotions)
    print("Cuvanje modela...")
    fishface.save("trained_emoclassifier.xml")
    print("Model sacuvan!")


