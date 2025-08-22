import math

def calculate_bounding_box(lat, lon, radius):
    # Earth's radius in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Radius of the bounding box in radians
    rad_dist = radius / R

    # Calculate min and max latitude
    min_lat = lat_rad - rad_dist
    max_lat = lat_rad + rad_dist

    # Calculate min and max longitude
    min_lon = lon_rad - rad_dist / math.cos(lat_rad)
    max_lon = lon_rad + rad_dist / math.cos(lat_rad)

    # Convert the results back to degrees
    min_lat = math.degrees(min_lat)
    max_lat = math.degrees(max_lat)
    min_lon = math.degrees(min_lon)
    max_lon = math.degrees(max_lon)

    return max_lat, min_lat, max_lon, min_lon

if __name__ == "__main__":
    lat = float(input("Enter the latitude of the center: "))
    lon = float(input("Enter the longitude of the center: "))
    radius = float(input("Enter the radius in kilometers: "))

    max_lat, min_lat, max_lon, min_lon = calculate_bounding_box(lat, lon, radius)

    print(f"Bounding Box:\nMax Latitude: {max_lat}\nMin Latitude: {min_lat}\nMax Longitude: {max_lon}\nMin Longitude: {min_lon}")
41.972718160591874,
41.79285383940813,
-87.52348959218492,
-87.7650764078151,