import os
import cv2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
abs_path = os.path.abspath(DATA_DIR)
print(f"Attempting to save images to: {abs_path}")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_path = os.path.join(DATA_DIR, str(j), f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        print(f"Saved image: {img_path}")

        counter += 1

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    files = os.listdir(class_dir)
    print(f"Number of images in class {j}: {len(files)}")

cap.release()
cv2.destroyAllWindows()