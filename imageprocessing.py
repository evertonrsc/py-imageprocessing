import cv2
import os
import requests
import sys
from google import genai

with open("../googleai.key", "r") as key_file:
    API_KEY = key_file.readline().strip()

client = genai.Client(api_key=API_KEY)
genai_model = "gemini-2.5-flash-lite"

def generate_image_urls(numimages):
    image_urls = []

    try:
        while len(image_urls) < numimages:
            generation_prompt = (
                "Generate {n} public domain image URL (either JPEG or PNG format) from "
                "trusted public domain image repositories, except any Wikimedia-related ones."
                "The URL must directly point to a valid image file ending with .jpg or .png "
                "and the file size must be less than 200 KB. Provide the final image URLs in "
                "plain text."
            ).format(n=numimages)
            generation_response = client.models.generate_content(
                model=genai_model,
                contents=generation_prompt
            )

            extraction_prompt = (
                "Extract all URLs from the following contents into a plain text list. "
                "Each URL must be on a new line. These are the contents: {text}"
            ).format(text=generation_response.text)
            extraction_response = client.models.generate_content(
                model=genai_model,
                contents=extraction_prompt
            )
            
            urls = extraction_response.text.strip().split('\n')
            for url in urls:
                url = url.strip()
                if is_accessible(url):
                    image_urls.append(url)
                if len(image_urls) == numimages:
                    break

        return image_urls
    except genai.errors.ServerError as err:
        print(f"Error: {err.message}")
        sys.exit(1)


def is_accessible(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except (requests.RequestException, ValueError) as e:
        print(f"Error accessing {url}: {e}")
        return False


def download_image(url, filename):
    useragent = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=useragent)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as image_file:
            image_file.write(response.content)
    else:
        print(f"Error while downloading image from {url}")
        sys.exit(1)


def to_grayscale(input_image, output_image):
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: could not read {input_image}")
        return
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    cv2.imwrite(output_image, grayscale)


if __name__ == "__main__":
    image_urls = generate_image_urls(1)

    for i in range(len(image_urls)):
        imagefile = "../images" + os.sep + str(i+1) + os.path.splitext(image_urls[i])[1]
        download_image(image_urls[i], imagefile)
        grayfile = "../gs-images" + os.sep + str(i+1) + os.path.splitext(image_urls[i])[1]
        to_grayscale(imagefile, grayfile)