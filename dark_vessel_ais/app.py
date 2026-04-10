import gradio as gr
import folium
import tempfile
import os
from datetime import datetime
import cv2

from detector import VesselDetector
from utils import get_detection_centers, match_detections_to_ais
from ais_fetcher import fetch_ais_data
from satellite import get_satellite_image

detector = VesselDetector("../Models/dark_vessel_best.pt", conf_thresh=0.25)

def analyze_location(lat, lon, date_str, buffer_km, use_sahi, radius_km, use_real_ais):
    """Process location and return report, map HTML, and preview image."""
    # Fetch satellite image
    img_path, thumb_url, bbox, capture_date = get_satellite_image(lat, lon, date_str, buffer_km)
    if img_path is None:
        return "❌ No satellite image found.", None, None
    
    # Detect vessels
    boxes, scores, (img_h, img_w) = detector.detect(img_path, use_sahi=use_sahi)
    if len(boxes) == 0:
        os.unlink(img_path)
        return "No vessels detected.", None, None
    
    # --- NEW: Draw bounding boxes on the image for the UI ---
    annotated_img = cv2.imread(img_path)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw Red boxes
    
    annotated_img_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
    cv2.imwrite(annotated_img_path, annotated_img)
    # --------------------------------------------------------

    # Convert to geographic coordinates
    centers = get_detection_centers(boxes, img_w, img_h, bbox)
    
    # Fetch AIS
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    if use_real_ais:
        ais_points = fetch_ais_data(bbox, target_date)
    else:
        from ais_fetcher import simulate_ais_points
        ais_points = simulate_ais_points(bbox, num_points=30)
    
    # Match and flag
    results = match_detections_to_ais(centers, ais_points, radius_km=radius_km)
    
    # Generate report
    dark_count = sum(1 for r in results if r['is_dark'])
    report = f"## 📊 Analysis Report\n"
    report += f"- **Image Date:** {capture_date}\n"
    report += f"- **Detected vessels:** {len(results)}\n"
    report += f"- **Dark vessels (no AIS):** {dark_count}\n\n"
    report += "### Vessel List\n"
    for i, r in enumerate(results):
        status = "🔴 **DARK**" if r['is_dark'] else "🟢 NORMAL"
        report += f"{i+1}. {status} at ({r['lat']:.5f}, {r['lon']:.5f})"
        if not r['is_dark']:
            report += f" – Matched MMSI {r['matched_ais']['mmsi']} ({r['distance_km']:.2f} km)"
        report += "\n"
    
    # Create map
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add AIS points
    for ais in ais_points:
        folium.Marker(
            [ais['lat'], ais['lon']],
            popup=f"MMSI: {ais['mmsi']}<br>Name: {ais['name']}",
            icon=folium.Icon(color='green', icon='ship', prefix='fa')
        ).add_to(m)
    
    # Add detections
    for r in results:
        color = 'red' if r['is_dark'] else 'orange'
        folium.CircleMarker(
            [r['lat'], r['lon']],
            radius=8,
            color=color,
            fill=True,
            popup=f"Detection at ({r['lat']:.5f}, {r['lon']:.5f})"
        ).add_to(m)
    
    map_path = tempfile.NamedTemporaryFile(suffix='.html', delete=False).name
    m.save(map_path)
    
    # Cleanup
    os.unlink(img_path)
    
    # RETURN THE ANNOTATED IMAGE INSTEAD OF THE THUMBNAIL URL
    return report, map_path, annotated_img_path

def process_uploaded_image(img_path, use_sahi):
    """Takes a user-uploaded image, runs YOLO, and draws boxes."""
    if img_path is None:
        return "⚠️ Please upload an image.", None
        
    print("📸 Scanning manual upload...")
    
    # 1. Run YOLO detection
    boxes, scores, (img_h, img_w) = detector.detect(img_path, use_sahi=use_sahi)
    
    # 2. Draw bright red boxes around the ships
    annotated_img = cv2.imread(img_path)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3) # 3 is thickness
        
    # 3. Save the annotated image to a temporary file
    out_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
    cv2.imwrite(out_path, annotated_img)
    
    # 4. Generate a quick text report
    report = f"### ✅ Manual Scan Complete\n"
    report += f"**Detected {len(boxes)} vessels in the uploaded image.**\n"
    
    return report, out_path

# Simple Gradio Interface
# TAB 1: Your original Map & AIS tool
map_tool = gr.Interface(
    fn=analyze_location,
    inputs=[
        gr.Number(label="Latitude", value=33.740),
        gr.Number(label="Longitude", value=-118.250),
        gr.Textbox(label="Date (YYYY-MM-DD)", value="2024-01-15"),
        gr.Slider(label="Search Radius (km)", minimum=1, maximum=50, value=10, step=1),
        gr.Checkbox(label="Use SAHI (tiled inference)", value=True),
        gr.Slider(label="AIS Matching Radius (km)", minimum=0.1, maximum=5.0, value=1.0, step=0.1),
        gr.Checkbox(label="Use Real AIS Data", value=False)
    ],
    outputs=[
        gr.Markdown(label="Detection Report"),
        gr.HTML(label="Interactive Map"),
        gr.Image(label="Satellite Pass")
    ]
)

# TAB 2: The new Manual Image Upload tool
upload_tool = gr.Interface(
    fn=process_uploaded_image,
    inputs=[
        gr.Image(type="filepath", label="Upload Satellite Image (.jpg or .png)"),
        gr.Checkbox(label="Use SAHI (Recommended for tiny ships)", value=True)
    ],
    outputs=[
        gr.Markdown(label="Scan Results"),
        gr.Image(label="Annotated Image")
    ]
)

# Combine them into a Tabbed App!
app = gr.TabbedInterface(
    [map_tool, upload_tool], 
    ["🌍 Live Map Search", "📸 Manual Image Scan"],
    title="Dark Vessel Detection System"
)

if __name__ == "__main__":
    app.launch(share=True)