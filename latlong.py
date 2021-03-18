from geopy.geocoders import Nominatim


def get_co(add):
    geolocator = Nominatim(user_agent="latlong")
    location = geolocator.geocode(add)

    return location.latitude, location.longitude


if __name__ == '__main__':
    lat, long = get_co('lajpat nagar Delhi')
    print(lat)
    print(long)
