import ee
import geemap
import datetime
import tempfile
import os

def initialize_gee():
    """Initialize Earth Engine. Run `earthengine authenticate` once first."""
    try:
        ee.Initialize()
    except Exception:
        print("Earth Engine not initialized. Run: earthengine authenticate")
        raise

def get_satellite_image(lat, lon, date_str, buffer_km=10):
    """
    Fetch a recent Sentinel-2 image for a point and date.
    Returns: (image_path, thumbnail_url, bbox, capture_date)
    """
    initialize_gee()
    
    # Parse date
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start_date = (date - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    end_date = (date + datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    
    # Region of interest (square buffer)
    region = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000).bounds()
    
    # Get bounding box coordinates
    coords = region.coordinates().getInfo()[0]
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Load Sentinel-2 surface reflectance
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    # Get first image
    image = collection.first()
    if image is None:
        return None, None, bbox, None
    
    # Get capture date
    capture_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    
    # Download as GeoTIFF
    output_path = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
    geemap.download_ee_image(image, filename=output_path, region=region, scale=10)
    
    # Thumbnail URL for preview
    url = image.getThumbURL({
        'min': 0, 'max': 3000, 
        'bands': ['B4', 'B3', 'B2'], 
        'dimensions': 512, 
        'format': 'png'
    })
    
    return output_path, url, bbox, capture_date