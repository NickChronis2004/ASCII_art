import cv2
import numpy as np
import time
import os
import platform

# ASCII χαρακτήρες για αναπαράσταση φωτεινότητας
ASCII_CHARS = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@', '1', '4']

def clear_terminal():
    """Καθαρισμός τερματικού ανάλογα με το λειτουργικό σύστημα"""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def resize_image(image, new_width=100):
    """Αλλαγή μεγέθους εικόνας με διατήρηση αναλογιών"""
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(new_width * aspect_ratio * 0.55)
    return cv2.resize(image, (new_width, new_height))

def to_grayscale(image):
    """Μετατροπή εικόνας σε grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def map_pixel_to_char(pixel_value):
    """Αντιστοίχιση τιμής pixel σε χαρακτήρα ASCII"""
    index = pixel_value * (len(ASCII_CHARS) - 1) // 255
    return ASCII_CHARS[index]

def pixels_to_ascii(image):
    """Μετατροπή όλων των pixels σε ASCII χαρακτήρες"""
    pixels = image.flatten()
    ascii_image = [map_pixel_to_char(p) for p in pixels]
    return ''.join(ascii_image)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam failed")
        return

    ascii_width = 150
    SHOW_GUI = True  # ➜ Άλλαξε το σε True αν θέλεις GUI παράθυρα με OpenCV

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Webcam failed")
                break

            frame = cv2.flip(frame, 1)
            resized_frame = resize_image(frame, ascii_width)
            gray_frame = to_grayscale(resized_frame)

            # Βελτίωση αντίθεσης + ανίχνευση περιγραμμάτων
            gray_frame = cv2.equalizeHist(gray_frame)
            gray_frame = cv2.convertScaleAbs(gray_frame, alpha=1.5, beta=0)
            edges = cv2.Canny(gray_frame, 100, 200)
            gray_frame = cv2.addWeighted(gray_frame, 0.8, edges, 0.5, 0)

            ascii_frame = pixels_to_ascii(gray_frame)

            height, width = gray_frame.shape
            ascii_lines = [ascii_frame[i:i + width] for i in range(0, len(ascii_frame), width)]
            ascii_image = '\n'.join(ascii_lines)

            # Καθαρισμός και εμφάνιση στο terminal (cross-platform)
            clear_terminal()
            print(ascii_image)

            if SHOW_GUI:
                # Εμφάνιση του πρωτότυπου frame (προαιρετικά)
                cv2.imshow('Webcam Original', frame)

                # Δημιουργία ASCII καμβά με OpenCV
                font_size = 0.35
                line_height = 12
                char_width = 6
                ascii_canvas = np.zeros((gray_frame.shape[0] * line_height,
                                         gray_frame.shape[1] * char_width, 3), dtype=np.uint8)

                for i, line in enumerate(ascii_lines):
                    y_pos = (i + 1) * line_height
                    cv2.putText(ascii_canvas, line, (5, y_pos),
                                cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 1)

                cv2.imshow('ASCII Art', ascii_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
