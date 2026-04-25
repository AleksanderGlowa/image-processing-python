import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

def load_image_from_url(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    resp = urllib.request.urlopen(req)
    image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

url = "https://upload.wikimedia.org/wikipedia/commons/e/e4/Lemon.jpg"
image = load_image_from_url(url)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite("wynik_gray_rotated.jpg", rotated)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Oryginal")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(rotated, cmap="gray")
plt.title("Po obrobce")
plt.axis("off")
plt.show()

print("Macierz obrazu (fragment):")
print(rotated[:10, :10]) 