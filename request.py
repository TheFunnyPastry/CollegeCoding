import requests

# URL of the file
url = 'https://www.cs.fsu.edu/~liux/courses/deepRL/assignments/zip_test.txt'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the content to a file
    with open('zip_test.txt', 'wb') as file:
        file.write(response.content)
    print('File downloaded and saved as zip_test.txt')
else:
    print(f'Failed to download file. Status code: {response.status_code}')
