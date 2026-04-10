# Dark Vessel Detection
# 🚢 Dark Vessel Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Project Overview
The **Dark Vessel Detection System** is an end-to-end Machine Learning pipeline designed to locate ships conducting illegal activities (such as illegal, unreported, and unregulated fishing or smuggling). These vessels often disable their Automatic Identification System (AIS) transponders to "go dark" and avoid authorities. 

This system solves the problem by fusing **Computer Vision (YOLOv8)** with **Geospatial Data**. It scans open-source satellite imagery to physically locate ships, and cross-references those GPS coordinates with AIS tracking data. If a ship exists on camera but has no matching AIS ping, it is immediately flagged as a **Dark Vessel**.

## ✨ Key Features
* **Dual-Tab Interface:** Built with Gradio for seamless user interaction.
* **Live Map Search:** Input geographic coordinates and a date to automatically fetch satellite passes, detect ships, and plot them on an interactive Folium map.
* **Manual Override Scan:** A fallback tool allowing investigators to upload private drone or satellite footage for instant computer vision scanning.
* **Open-Source Satellite Data:** Integrated with the **Microsoft Planetary Computer STAC API** to fetch free Sentinel-2 imagery, bypassing expensive commercial or highly-restricted platforms.
* **SAHI Integration:** Utilizes Slicing Aided Hyper Inference (SAHI) to detect microscopic ship footprints in massive high-resolution satellite swaths.

## System Architecture
1. **Data Ingestion:** Fetches Sentinel-2 optical imagery based on user-defined Bounding Boxes (BBox) and timestamps.
2. **AI Inference:** YOLOv8 model scans the imagery and outputs bounding boxes.
3. **Geospatial Conversion:** Pixel coordinates are mapped to real-world Latitude/Longitude coordinates.
4. **AIS Cross-Referencing:** Detected coordinates are matched against AIS data within a defined proximity radius.
5. **Visualization:** Results are compiled into a comprehensive text report and a Folium HTML map (Red dots = Dark Vessels, Green dots = Compliant Vessels).

## Data Strategy & Proof of Concept (PoC)
Global, historical AIS data is typically locked behind commercial licenses costing upwards of $10,000. To demonstrate the architecture's viability without commercial API keys, this project utilizes a **Simulated Data Stream** for PoC purposes. 

The geographic matching algorithm mathematically proves that *if* a commercial AIS stream is connected, the cross-referencing logic successfully separates legal vessels from dark vessels. The system is also designed to be easily extensible to read open-source US Coast Guard (MarineCadastre) CSV files for localized historical analysis.

## Deployment
This model has been deployed at [https://huggingface.co/spaces/cabralchege/Optical-Dark-Vessel-Detection]
## Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/dark-vessel-detection.git](https://github.com/yourusername/dark-vessel-detection.git)
cd dark-vessel-detection
