import urllib.request

def check_internet(url="http://www.google.com"):
    try:
        response = urllib.request.urlopen(url, timeout=5)
        print("Internet access is available.")
        return True
    except Exception as e:
        print("No internet access:", e)
        return False

check_internet()
