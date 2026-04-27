# rpi5-ai-vision

## Opis projekta
### Diplomski rad: Detekcija ljudi u realnom vremenu na Raspberry Pi 5 platformi

Ovaj projekt fokusira se na implementaciju, evaluaciju i optimizaciju suvremenih računalnih modela za detekciju objekata (ljudi) na **Raspberry Pi 5** računalu. Cilj rada je istražiti granice performansi *edge* uređaja koristeći različite arhitekture modela i tehnike kvantizacije.

## Tehnološki Stack

* **Jezik:** Python 3.11
* **ML Frameworks:** ONNX Runtime, TFLite (TensorFlow Lite), Ultralytics (YOLO)
* **Computer Vision:** OpenCV
* **Backend:** FastAPI / Flask (Web Server & Streaming)
* **Frontend:** HTML5, JavaScript (WebSockets, Chart.js)
* **Hardware Control:** GPIOZero, lgpio (PIR senzor integracija)

## Hardverska Konfiguracija

Projekt se izvodi na maksimalno optimiziranom hardverskom setupu:

| Komponenta | Specifikacija |
| :--- | :--- |
| **Računalo** | Raspberry Pi 5 (8GB RAM) |
| **Pohrana** | M.2 HAT+ s 256GB NVMe SSD (za maksimalan I/O throughput) |
| **Kamera** | Camera Module 3 NoIR (podržava infracrveni spektar za noćni rad) |
| **Senzor** | PIR senzor (pasivni infracrveni senzor za trigger detekcije) |

## 📊 Pregled Modela i Evaluacija

Središnji dio projekta je usporedba različitih obitelji modela kroz tri razine preciznosti (**FP32**, **FP16**, **INT8**).

### Modeli u testiranju:
* **YOLOv8 & YOLOv10 (n/s):** Industrijski standard za balans brzine i točnosti.
* **RT-DETR (Real-Time DEtection TRansformer):** Napredna arhitektura bazirana na transformerima.
* **MobileNet-SSD v2 / EfficientDet-Lite:** Lagani modeli optimizirani za mobilne uređaje.

### Metrike koje pratimo:
* **Preciznost:** mAP@0.5, mAP@0.5:0.95, Precision, Recall.
* **Performanse:** FPS (Sličice u sekundi), Latencija (ms).
* **Resursi:** CPU opterećenje (%), RAM potrošnja (MB), Temperatura procesora (°C).
* **Efikasnost:** mAP po FPS-u, mAP po veličini modela (MB).

## 🚀 Arhitektura Sustava

Projekt je podijeljen u tri glavna segmenta:

1.  **Inference Engine:** Obrada video streama u realnom vremenu, primjena NMS (*Non-Maximum Suppression*) i iscrtavanje bounding boxova.
2.  **Web Dashboard:** Live stream putem WebSocketa s prikazom statistike performansi u realnom vremenu (grafovi latencije i FPS-a).
3.  **Power Management:** Integracija PIR senzora koji aktivira sustav detekcije samo kada se detektira kretanje (ušteda energije i resursa).

## Autor
Eric Jakovac

## Datum kreiranja
27.04.2026.
