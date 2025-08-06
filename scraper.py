import requests
from bs4 import BeautifulSoup

def scrape_site(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ')
    with open('scraped_data.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("✅ Scraped content saved to scraped_data.txt")

if __name__ == "__main__":
    scrape_site("https://realpython.com/python-web-scraping-practical-introduction/")
