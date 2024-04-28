import face_recognition
import cv2
import numpy as np
import os
import torch

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

def detect_objects(frame):
    results = model(frame)
    return results

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Criar arrays de conhecidos encodings de rostos e seus nomes
known_face_encodings = []
known_face_names = []

# Carregar cada imagem do diretório 'registered_faces'
for filename in os.listdir("registered_faces"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join("registered_faces", filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Pegar uma referência para a webcam
video_capture = cv2.VideoCapture(10)

while True:
    # Capturar frame a frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Processar apenas a cada outro frame do vídeo para economizar tempo
    if process_this_frame:
        # Detecção de rostos
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "INTRUSO"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    # Detecção de objetos com YOLOv5
    results = detect_objects(frame)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # Exibir resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Desenhar retângulos e nomes para rostos
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    for label, cord in zip(labels, cords):
        x1, y1, x2, y2, *_ = cord
        x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, model.names[int(label)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    process_this_frame = not process_this_frame



    process_this_frame = not process_this_frame

    # Exibir o vídeo resultante
    cv2.imshow('AI Recognition', frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e destruir todas as janelas
video_capture.release()
cv2.destroyAllWindows()
