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
    
    return report, map_path, thumb_url

# Simple Gradio Interface
iface = gr.Interface(
    fn=analyze_location,
    inputs=[
        gr.Number(label="Latitude", value=34.0),
        gr.Number(label="Longitude", value=-119.0),
        gr.Textbox(label="Date (YYYY-MM-DD)", value="2024-01-01"),
        gr.Slider(label="Search Radius (km)", minimum=1, maximum=50, value=10, step=1),
        gr.Checkbox(label="Use SAHI (tiled inference)", value=True),
        gr.Slider(label="AIS Matching Radius (km)", minimum=0.1, maximum=5.0, value=1.0, step=0.1),
        gr.Checkbox(label="Use Real AIS Data", value=False),  # Set to False for testing
    ],
    outputs=[
        gr.Markdown(label="📈 Detection Report"),
        gr.HTML(label="🗺️ Interactive Map"),
        gr.Image(label="📸 Satellite Preview", type="filepath"),
    ],
    title="🛰️ Dark Vessel Detector",
    description="Pull satellite imagery, detect vessels, cross-reference with AIS to identify dark vessels.",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7861, share=True)