import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/8/8c/Cristiano_Ronaldo_2018.jpg"

def load_remote_image(url: str) -> np.ndarray:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    data = np.frombuffer(response.content, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Nie udało się zdekodować obrazu.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def plot_histograms(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Zdjęcie")
    axes[0, 0].axis("off")

    axes[0, 1].plot(hist_gray, color="black")
    axes[0, 1].set_title("Histogram całego zdjęcia")
    axes[0, 1].set_xlim([0, 255])

    axes[1, 0].plot(hist_r, color="red", label="R")
    axes[1, 0].plot(hist_g, color="green", label="G")
    axes[1, 0].plot(hist_b, color="blue", label="B")
    axes[1, 0].set_title("Histogram RGB")
    axes[1, 0].set_xlim([0, 255])
    axes[1, 0].legend()

    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

def evaluate_image_quality(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    total = gray.size

    shadow_clip = float(np.sum(gray <= 2) / total)
    highlight_clip = float(np.sum(gray >= 253) / total)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dynamic_range = p95 - p5

    score = 100.0

    score -= min(40.0, shadow_clip * 1500.0)
    score -= min(40.0, highlight_clip * 1500.0)

    if std_brightness < 35:
        score -= (35 - std_brightness) * 0.8

    if dynamic_range < 120:
        score -= (120 - dynamic_range) * 0.5

    if mean_brightness < 80 or mean_brightness > 180:
        score -= 10.0

    score = float(np.clip(score, 0, 100))

    if shadow_clip > 0.03 and highlight_clip > 0.03:
        diagnosis = "Zdjęcie ma utratę detali w cieniach i światłach."
    elif shadow_clip > 0.03:
        diagnosis = "Zdjęcie jest niedoświetlone."
    elif highlight_clip > 0.03:
        diagnosis = "Zdjęcie jest prześwietlone."
    elif std_brightness < 35 or dynamic_range < 120:
        diagnosis = "Zdjęcie ma niski kontrast."
    else:
        diagnosis = "Zdjęcie jest poprawnie naświetlone."

    return {
        "score": score,
        "diagnosis": diagnosis,
        "shadow_clip": shadow_clip,
        "highlight_clip": highlight_clip
    }

def improve_image(img_rgb: np.ndarray):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))

    improved_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return improved_rgb

img = load_remote_image(IMAGE_URL)

plot_histograms(img)

info = evaluate_image_quality(img)

print("Wynik jakości:", round(info["score"], 2), "/ 100")
print("Diagnoza:", info["diagnosis"])
print("Clipping cieni:", round(info["shadow_clip"] * 100, 2), "%")
print("Clipping świateł:", round(info["highlight_clip"] * 100, 2), "%")

improved = improve_image(img)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].imshow(img)
ax[0].set_title("Oryginał")
ax[0].axis("off")

ax[1].imshow(improved)
ax[1].set_title("Po poprawie")
ax[1].axis("off")

plt.tight_layout()
plt.show()