# simple server test
import requests
import json 

def main():
    BASE_URL = "http://localhost:5000/pose"
    result = requests.get(BASE_URL)
    json_response = json.loads(result.text)
    print(json_response)

if __name__ == "__main__":
    main()