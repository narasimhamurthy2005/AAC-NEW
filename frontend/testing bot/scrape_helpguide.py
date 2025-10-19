import requests
from bs4 import BeautifulSoup
import os

# List of HelpGuide.org pages to scrape
urls = [
    'https://www.helpguide.org/mental-health/adhd',
    'https://www.helpguide.org/mental-health/addiction',
    'https://www.helpguide.org/mental-health/anxiety',
    'https://www.helpguide.org/mental-health/depression',
    'https://www.helpguide.org/aging/healthy-aging',
    'https://www.helpguide.org/mental-health/meditation',
    'https://www.helpguide.org/relationships/communication',
    'https://www.helpguide.org/mental-health/emotional-intelligence',
    'https://www.helpguide.org/relationships/social-connection',
    'https://www.helpguide.org/relationships/sexual-health',
    'https://www.helpguide.org/relationships/domestic-abuse',
    'https://www.helpguide.org/wellness/fitness',
    'https://www.helpguide.org/wellness/nutrition',
    'https://www.helpguide.org/wellness/sleep',
    'https://www.helpguide.org/mental-health/stress',
    'https://www.helpguide.org/mental-health/wellbeing',
    'https://www.helpguide.org/wellness/career',
    'https://www.apa.org/topics/stress'
]

# Create a folder to save scraped text files
os.makedirs("scraped_pages", exist_ok=True)

for url in urls:
    if not url.strip():  # Skip empty URLs
        print("Skipped empty URL")
        continue
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        continue
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract all paragraph text
    content = ' '.join([p.get_text() for p in soup.find_all('p')])
    
    # Generate a valid file name from the URL
    page_name = url.strip('/').split('/')[-1]
    filename = os.path.join("scraped_pages", f"{page_name}.txt")
    
    # Save the content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Saved content to {filename}")
