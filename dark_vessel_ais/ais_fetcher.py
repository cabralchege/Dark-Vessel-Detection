import aiohttp
import asyncio
import nest_asyncio
from datetime import datetime, timedelta

nest_asyncio.apply()

async def fetch_ais_data_async(bbox, start_time, end_time):
    """
    Fetch real AIS data from aisstream.io.
    bbox: (min_lon, min_lat, max_lon, max_lat)
    Returns list of AIS points with lat, lon, mmsi, name
    """
    polygon = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[0], bbox[1]]
    ]
    
    payload = {
        "APIKey": "079b9ad087f3fe3c06d973ce3608a47b2f6a2e6b",
        "BoundingBox": polygon,
        "TimeStart": start_time,
        "TimeEnd": end_time,
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post('https://api.aisstream.io/v1/positions', json=payload, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    features = data.get('Features', [])
                    ais_points = []
                    for feat in features:
                        coords = feat.get('geometry', {}).get('coordinates', [])
                        props = feat.get('properties', {})
                        if len(coords) >= 2:
                            ais_points.append({
                                'lon': coords[0],
                                'lat': coords[1],
                                'mmsi': props.get('mmsi', 'unknown'),
                                'name': props.get('shipname', 'Unknown'),
                                'speed': props.get('speed_over_ground', 0),
                                'course': props.get('course_over_ground', 0)
                            })
                    return ais_points
                else:
                    print(f"AIS API error: {resp.status}")
                    return []
        except Exception as e:
            print(f"AIS fetch error: {e}")
            return []

def fetch_ais_data(bbox, target_date):
    """Synchronous wrapper for AIS fetch."""
    start_time = (target_date - timedelta(hours=6)).isoformat() + 'Z'
    end_time = (target_date + timedelta(hours=6)).isoformat() + 'Z'
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(fetch_ais_data_async(bbox, start_time, end_time))