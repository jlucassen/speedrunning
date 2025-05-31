import requests
from bs4 import BeautifulSoup
import json
import os
import tqdm

def print_pin_urls(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    imgs = soup.find_all('img', attrs={'data-pin-url': True})
    for img in imgs:
        yield img['data-pin-url']

def download_image(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    return False

def get_processed_urls(jsonl_path):
    processed_urls = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'url' in data:
                        processed_urls.add(data['url'])
                except json.JSONDecodeError:
                    continue
    return processed_urls

def scrape_meal(url, output_dir='homechef', jsonl_path='homechef.jsonl'):
    # Check if URL has already been processed
    processed_urls = get_processed_urls(jsonl_path)
    if url in processed_urls:
        print(f"Skipping already processed URL: {url}")
        return

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    og_img = soup.find('meta', property='og:image')
    img_url = None
    if og_img:
        img_url = og_img.get('content')
    if not img_url:
        main_img = soup.find('img', {'itemprop': 'image'})
        if main_img and main_img.get('src'):
            img_url = main_img['src']
    if not img_url:
        print('No image found!')
        return

    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_filename = os.path.join(output_dir, os.path.basename(img_url.split('?')[0]))
    
    # Download the image
    download_image(img_url, img_filename)

    # Nutrition extraction: look for label-value pairs
    nutrition = {}
    keywords = ['calories', 'carbohydrates', 'net carbs', 'fat', 'protein', 'sodium']
    for tag in soup.find_all(string=True):
        text = tag.strip().lower()
        for key in keywords:
            if text == key:
                # Try next siblings for the value
                value = None
                sib = tag.parent.find_next_sibling()
                # Sometimes the value is in a <strong> or <b> or just the next tag
                if sib:
                    value = sib.get_text(strip=True)
                else:
                    # Try next element in the parent
                    next_el = tag.find_next(string=True)
                    if next_el and next_el.strip().lower() != key:
                        value = next_el.strip()
                if value:
                    nutrition[key] = value
    nutrition['image_path'] = img_filename
    nutrition['url'] = url  # Add URL to the saved data
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(nutrition, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    for url in tqdm.tqdm(list(print_pin_urls('https://www.homechef.com/our-menu'))):
        scrape_meal(url)
