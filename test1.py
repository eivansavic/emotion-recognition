import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Definisanje redosleda emocija
participants = glob.glob("source_emotion/*",  recursive=True)  # Vraca se lista svih forldera sa rednim brojem

for x in participants:
    part = "%s" % x[-4:]  # Cuvamo trenutni broj foldera
    for sessions in glob.glob("%s/*" % x):  # Prolazimo kroz podforldere trenutnog glavnog foldera
        for files in glob.glob("%s/*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            emotion = int(
                float(file.readline()))  # emocije su kodirane kao float vrednosti, potrebno ih je konvertovati u int

            sourcefile_emotion = glob.glob("source_images/%s/%s/*" % (part, current_session))[-1]  # uzimamo putanju poslednje slike koja sadrzi emociju
            sourcefile_neutral = glob.glob("source_images/%s/%s/*" % (part, current_session))[0]  # isto radimo i za neutralne slike

            dest_neut = "sorted_set/neutral/%s" % sourcefile_neutral[25:]  # Generisemo putanju za upis neutralne slike
            dest_emot = "sorted_set/%s/%s" % (emotions[emotion], sourcefile_emotion[25:])  # Isto radimo za slike sa emocijama

            copyfile(sourcefile_neutral, dest_neut)  # Kopiranje fajla
            copyfile(sourcefile_emotion, dest_emot)