import os
import cv2
import numpy as np
from keras.models import load_model


model_path = 'best_model.h5'
model = load_model(model_path)

#Ändern des Ein und Ausgabeordners
data_dir = '\\Ordner'
output_dir = '\\Ordner'


label_dict = {'Bicycle': 0, 'Bus': 1, 'Car': 2, 'Crosswalk': 3, 'Hydrant': 4, 
              'Motorcycle': 5, 'Palm': 6, 'Stair': 7, 'Traffic Light': 8}

#labelnummern=namen
reverse_label_dict = {v: k for k, v in label_dict.items()}


def draw_boxes(image, boxes, font_scale=0.3, font=cv2.FONT_HERSHEY_SIMPLEX):
    for box in boxes:
       
        box = [int(coord * image.shape[1]) if i > 0 else coord for i, coord in enumerate(box)]
        label = int(box[0])
        xmin, ymin, xmax, ymax = box[1:]

        # Klassennamen 
        class_name = reverse_label_dict.get(label, "Unknown")

        # Begrenzungsrahmen 
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, class_name, (xmin, ymin - 10), font, font_scale, (36,255,12), 1)
    return image


def load_and_predict(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.png'):
            image_path = os.path.join(data_dir, file)
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (120, 120))

           
            prediction = model.predict(np.array([resized_image / 120.0]))[0]

         
            print(f"Raw prediction for {file}: {prediction}")
         
            num_boxes = len(prediction) // 5
            boxes = [prediction[i*5:(i+1)*5] for i in range(num_boxes)]

            boxed_image = draw_boxes(image, boxes)

            #Save und Anzeigen
            boxed_image_path = os.path.join(output_dir, 'boxed_' + file)  
            cv2.imwrite(boxed_image_path, boxed_image)
            cv2.imshow('Object Detection', boxed_image)
            cv2.waitKey(0)

load_and_predict(data_dir)
cv2.destroyAllWindows()