from geopy.distance import geodesic

def pixel_to_geo(x, y, img_w, img_h, bbox):
    """
    Convert pixel coordinates to geographic lon/lat.
    bbox = (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + (x / img_w) * (max_lon - min_lon)
    lat = max_lat - (y / img_h) * (max_lat - min_lat)  # y=0 is top
    return lon, lat

def get_detection_centers(boxes, img_w, img_h, bbox):
    """Convert each detection box to its center point in geographic coordinates."""
    centers = []
    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        lon, lat = pixel_to_geo(x_center, y_center, img_w, img_h, bbox)
        centers.append((lon, lat, box))
    return centers

def match_detections_to_ais(detection_centers, ais_points, radius_km=1.0):
    """
    For each detection, find nearest AIS point within radius.
    Returns list of dicts with detection info and match status.
    """
    results = []
    for lon, lat, box in detection_centers:
        nearest_ais = None
        min_dist = float('inf')
        for ais in ais_points:
            dist = geodesic((lat, lon), (ais['lat'], ais['lon'])).km
            if dist < min_dist:
                min_dist = dist
                nearest_ais = ais
        is_dark = (min_dist > radius_km)
        results.append({
            "box": box,
            "lon": lon,
            "lat": lat,
            "matched_ais": nearest_ais if not is_dark else None,
            "distance_km": min_dist,
            "is_dark": is_dark
        })
    return results