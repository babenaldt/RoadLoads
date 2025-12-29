
import csv
import math
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np

def parse_gpx(filepath):
    """
    Parses a GPX file and returns a list of (time, lat, lon, elevation) tuples.
    Handles missing time fields.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Define namespace (GPX files usually have one)
    # Try to detect namespace or handle both
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    points = []
    
    # Find all track points (handle with and without namespace)
    # Some GPX files use 1.0, some 1.1
    track_points = root.findall('.//gpx:trkpt', ns)
    if not track_points:
        # Try without namespace
        track_points = root.findall('.//trkpt')
        ns = {} # Clear namespace for subsequent finds
    
    for trkpt in track_points:
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        
        ele_node = trkpt.find('gpx:ele', ns) if ns else trkpt.find('ele')
        ele = float(ele_node.text) if ele_node is not None else 0.0
        
        time_node = trkpt.find('gpx:time', ns) if ns else trkpt.find('time')
        time = None
        
        if time_node is not None and time_node.text:
            time_str = time_node.text
            try:
                time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                try:
                    time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    pass # Keep as None
        
        # Look for maxspeed in extensions
        maxspeed = None
        extensions = trkpt.find('gpx:extensions', ns) if ns else trkpt.find('extensions')
        if extensions is not None:
            # Search recursively for maxspeed tag, ignoring namespace
            for child in extensions.iter():
                if 'maxspeed' in child.tag:
                    try:
                        maxspeed = float(child.text)
                        break
                    except (ValueError, TypeError):
                        pass

        points.append({
            'time': time,
            'lat': lat,
            'lon': lon,
            'ele': ele,
            'maxspeed': maxspeed
        })
        
    return points

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def convert_gpx_to_cycle(gpx_filepath, output_filepath, target_speed_mph=None, smoothing_window=5, maxspeed_units='mph', max_accel=1.5):
    """
    Converts GPX data to drive cycle CSV (time, speed, grade).
    """
    print(f"Parsing {gpx_filepath}...")
    try:
        points = parse_gpx(gpx_filepath)
    except Exception as e:
        print(f"Error parsing GPX: {e}")
        return

    if not points:
        print("No track points found in GPX.")
        return

    print(f"Found {len(points)} points.")
    
    # Calculate raw metrics
    data = []
    
    # Determine mode: Real Time or Synthetic Speed
    has_timestamps = points[0]['time'] is not None
    use_synthetic_speed = not has_timestamps
    
    if use_synthetic_speed:
        if target_speed_mph is None:
            # Check for maxspeed
            has_maxspeed = any(p.get('maxspeed') is not None for p in points)
            if not has_maxspeed:
                print("Error: GPX file has no timestamps and no --speed provided, and no maxspeed tags found.")
                return
            print(f"Using maxspeed data from GPX file (units: {maxspeed_units}).")
        else:
            print(f"Using constant target speed: {target_speed_mph} mph")

    target_speed_mps = (target_speed_mph * 0.44704) if target_speed_mph else 0
    current_synthetic_time = 0.0
    
    # Conversion factor for maxspeed
    maxspeed_factor = 0.44704 if maxspeed_units == 'mph' else 0.27778 # mph to m/s or km/h to m/s
    
    last_valid_maxspeed = 60.0 * 0.44704 # Default fallback if first point missing maxspeed
    
    for i in range(1, len(points)):
        p1 = points[i-1]
        p2 = points[i]
        
        dist = haversine_distance(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        ele_change = p2['ele'] - p1['ele']
        
        if use_synthetic_speed:
            # Calculate time based on target speed or maxspeed
            if target_speed_mph is not None:
                speed = target_speed_mps
            else:
                # Use maxspeed
                ms = p1.get('maxspeed')
                if ms is not None:
                    speed = float(ms) * maxspeed_factor
                    last_valid_maxspeed = speed
                else:
                    speed = last_valid_maxspeed
            
            if dist > 0:
                if speed < 0.1: speed = 10.0 # Avoid stuck
                dt = dist / speed
            else:
                dt = 0
                speed = 0
            
            current_synthetic_time += dt
            t_rel = current_synthetic_time
        else:
            # Use real time
            if p1['time'] is None or p2['time'] is None:
                continue # Skip points with missing time
                
            dt = (p2['time'] - p1['time']).total_seconds()
            if dt <= 0: continue
            
            speed = dist / dt # m/s
            t_rel = (p2['time'] - points[0]['time']).total_seconds()
        
        # Grade calculation (rise / run)
        if dist > 1.0:
            grade = (ele_change / dist) * 100
        else:
            grade = 0.0
            
        # Clamp grade to realistic values (-30% to +30%) to filter GPS noise
        grade = max(min(grade, 30.0), -30.0)
        
        data.append({
            'time': t_rel,
            'speed': speed,
            'grade': grade,
            'ele': p2['ele']
        })

    if not data:
        print("No valid segments found.")
        return

    # Interpolate to 1-second intervals
    print("Interpolating to 1Hz...")
    max_time = int(data[-1]['time'])
    new_times = np.arange(0, max_time + 1, 1)
    
    raw_times = [0] + [d['time'] for d in data]
    raw_speeds = [0] + [d['speed'] for d in data]
    raw_grades = [0] + [d['grade'] for d in data]
    
    interp_speeds = np.interp(new_times, raw_times, raw_speeds)
    interp_grades = np.interp(new_times, raw_times, raw_grades)
    
    # Smooth data (GPS speed/grade is noisy)
    print(f"Smoothing data (window={smoothing_window})...")
    
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    final_speeds = smooth(interp_speeds, smoothing_window)
    final_grades = smooth(interp_grades, smoothing_window * 2) # Grade needs more smoothing
    
    # Limit acceleration
    if max_accel is not None:
        print(f"Limiting acceleration to {max_accel} m/s^2...")
        
        # Force start from 0 to avoid initial spike from smoothing artifacts
        final_speeds[0] = 0.0
        
        for i in range(1, len(final_speeds)):
            # Limit positive acceleration
            if final_speeds[i] > final_speeds[i-1] + max_accel:
                final_speeds[i] = final_speeds[i-1] + max_accel
            # Limit deceleration (optional, but good for realism - set to -3 m/s^2)
            elif final_speeds[i] < final_speeds[i-1] - 3.0:
                final_speeds[i] = final_speeds[i-1] - 3.0

    # Write to CSV
    with open(output_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'speed', 'grade'])
        
        for i in range(len(new_times)):
            # Ensure non-negative speed
            s = max(0.0, final_speeds[i])
            g = final_grades[i]
            writer.writerow([new_times[i], f"{s:.2f}", f"{g:.2f}"])
            
    print(f"Successfully created {output_filepath}")
    print(f"Duration: {max_time} s")
    print(f"Max Speed: {max(final_speeds) * 2.23694:.1f} mph")
    print(f"Max Grade: {max(final_grades):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert GPX to Drive Cycle CSV')
    parser.add_argument('input', help='Input GPX file')
    parser.add_argument('output', help='Output CSV file')
    parser.add_argument('--speed', type=float, help='Target speed in MPH (for GPX files without time)')
    parser.add_argument('--smooth', type=int, default=5, help='Smoothing window size (default: 5)')
    parser.add_argument('--units', choices=['mph', 'kmh'], default='mph', help='Units for maxspeed tag in GPX (default: mph)')
    parser.add_argument('--max_accel', type=float, default=1.5, help='Maximum acceleration in m/s^2 (default: 1.5)')
    
    args = parser.parse_args()
    
    convert_gpx_to_cycle(args.input, args.output, args.speed, args.smooth, args.units, args.max_accel)
