import face_recognition
import cv2
import numpy as np
import os

# --- नए चेहरे जोड़ने के लिए सेटिंग्स ---
KNOWN_FACES_DIR = "known_faces"  # वह फोल्डर जहाँ आप ज्ञात चेहरों की तस्वीरें रखेंगे
# TOLERANCE = 0.6  # मिलान के लिए सहनशीलता (कम = अधिक सख्त)
MODEL = "hog"  # या "cnn" (cnn धीमा है लेकिन अधिक सटीक) - hog तेज है

print("ज्ञात चेहरों को लोड किया जा रहा है...")

known_face_encodings = []
known_face_names = []

# 'known_faces' फोल्डर से चेहरे लोड करें
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"चेतावनी: '{KNOWN_FACES_DIR}' फोल्डर नहीं मिला। कृपया इसे बनाएं और ज्ञात चेहरों की तस्वीरें डालें।")
else:
    for name in os.listdir(KNOWN_FACES_DIR):
        # फोल्डर के अंदर सब-फोल्डर को छोड़ दें (यदि कोई हो)
        if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name)):
            continue
        
        # इमेज फाइल का पूरा पाथ
        image_path = os.path.join(KNOWN_FACES_DIR, name)
        
        try:
            # इमेज लोड करें
            image = face_recognition.load_image_file(image_path)
            # फेस एन्कोडिंग प्राप्त करें (हम मानते हैं कि प्रत्येक तस्वीर में एक चेहरा है)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                # फाइल नाम से एक्सटेंशन हटाकर नाम के रूप में उपयोग करें
                known_face_names.append(os.path.splitext(name)[0])
            else:
                print(f"चेतावनी: '{name}' में कोई चेहरा नहीं मिला।")
        except Exception as e:
            print(f"चेतावनी: '{name}' को लोड करने में त्रुटि: {e}")

if not known_face_encodings:
    print("कोई ज्ञात चेहरा लोड नहीं हुआ। प्रोग्राम अज्ञात चेहरों को ही पहचानेगा।")
    # आप चाहें तो डिफ़ॉल्ट चेहरे यहां जोड़ सकते हैं यदि फोल्डर खाली है, जैसे मूल कोड में था
    # obama_image = face_recognition.load_image_file("obama.jpg")
    # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    # biden_image = face_recognition.load_image_file("biden.jpg")
    # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    # known_face_encodings = [obama_face_encoding, biden_face_encoding]
    # known_face_names = ["Barack Obama", "Joe Biden"]

print(f"{len(known_face_names)} ज्ञात चेहरे लोड किए गए: {known_face_names}")

# वेबकैम #0 (डिफ़ॉल्ट वाला) का रेफरेंस प्राप्त करें
video_capture = cv2.VideoCapture(0)

print("वेबकैम शुरू हो रहा है...")

while True:
    # वीडियो का एक फ्रेम पकड़ें
    ret, frame = video_capture.read()
    if not ret:
        print("वेबकैम से फ्रेम कैप्चर करने में विफल। बाहर निकल रहा है...")
        break

    # इमेज को BGR कलर (जो OpenCV उपयोग करता है) से RGB कलर (जो face_recognition उपयोग करता है) में बदलें
    # प्रदर्शन को बेहतर बनाने के लिए फ्रेम का आकार बदलें (वैकल्पिक)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1] # BGR to RGB

    # वीडियो के फ्रेम में सभी चेहरों और फेस एन्कोडिंग का पता लगाएं
    # मॉडल बदलने के लिए model="cnn" का उपयोग करें यदि आपने इसे ऊपर परिभाषित किया है
    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names_in_frame = []
    # वीडियो के इस फ्रेम में प्रत्येक चेहरे के माध्यम से लूप करें
    for face_encoding in face_encodings:
        name = "Unknown" # डिफ़ॉल्ट नाम
        if known_face_encodings: # यदि कोई ज्ञात चेहरा लोड किया गया हो
            # देखें कि क्या चेहरा ज्ञात चेहरे (चेहरों) से मेल खाता है
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding) #, tolerance=TOLERANCE)
            
            # ज्ञात चेहरे के साथ नए चेहरे की सबसे छोटी दूरी का उपयोग करें
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0: # सुनिश्चित करें कि face_distances खाली नहीं है
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
        
        face_names_in_frame.append(name)

    # परिणामों को प्रदर्शित करें (मूल फ्रेम पर क्योंकि small_frame बहुत छोटा है)
    for (top, right, bottom, left), name in zip(face_locations, face_names_in_frame):
        # स्केल बैक फेस लोकेशन क्योंकि हमने जिस फ्रेम का पता लगाया था, वह 1/4 आकार का था
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # चेहरे के चारों ओर एक बॉक्स बनाएं
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # चेहरे के नीचे नाम के साथ एक लेबल बनाएं
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # परिणामी इमेज प्रदर्शित करें
    cv2.imshow('Video', frame)

    # कीबोर्ड पर 'q' दबाकर बाहर निकलें!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# वेबकैम का हैंडल जारी करें
video_capture.release()
cv2.destroyAllWindows()
print("प्रोग्राम समाप्त।")
