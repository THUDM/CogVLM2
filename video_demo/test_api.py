import requests
url = 'http://127.0.0.1:5000/video_qa'
video_file = "test.mp4"
question = "Describe this video in detail."
temperature=0.2
files = {'video': open(video_file, 'rb')}
data = {'question': question,'temperature': temperature}
response = requests.post(url, files=files, data=data)
print(response.json()["answer"])
