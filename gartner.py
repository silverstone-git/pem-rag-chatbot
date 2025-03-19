from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
import random
import undetected_chromedriver as uc                                                                                                                                                      
driver = uc.Chrome(headless=False,use_subprocess=False)

# Random delays between actions
def human_delay():
    time.sleep(random.uniform(1, 3))

# Use this before each navigation
driver.get("https://www.gartner.com/en/insights")
human_delay()

# Wait for the dynamic content to load
wait = WebDriverWait(driver, 20)
try:
    print("waiting")
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'row.dynamic-content')))
    # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.row.dynamic-content')))
    print("waitdone")
except Exception as e:
    print("Timed out waiting for page to load")
    driver.quit()
    exit()

# Parse the page with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
main_div = soup.find('div', class_='row dynamic-content')
category_blocks = main_div.find_all('div', class_='category-block') if main_div else []

print("category blocks: ", category_blocks)

result = {}

for block in category_blocks:
    # Extract category name
    a_tag = block.find('a')
    if not a_tag:
        continue
    h4 = a_tag.find('h4', class_='categoryheadline')
    if not h4:
        continue
    category_name = h4.get_text(strip=True)
    result[category_name] = {}
    print("result", result)
    
    # Extract all article links in the category
    items_content = block.find('div', class_='categoryItemsContent')
    if not items_content:
        continue
    links = items_content.find_all('a', href=True)
    
    for link in links:
        href = link['href']
        article_title = link.get_text(strip=True)
        
        # Form absolute URL
        full_url = f'https://www.gartner.com{href}' if href.startswith('/') else href
        
        # Open article in a new tab
        original_window = driver.current_window_handle
        driver.switch_to.new_window('tab')
        try:
            human_delay()
            driver.get(full_url)
            human_delay()
            # Wait for article content to load
            article_wait = WebDriverWait(driver, 15)
            article_wait.until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
            
            # Parse article content
            # Modify the article content extraction part of your script:

            # Inside the article processing block after driver.get(full_url):
            article_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Extract article title
            title = article_soup.find('h1').get_text(strip=True) if article_soup.find('h1') else "Untitled"

            # Extract authors and date
            byline = ""
            authors = []
            date = ""
            byline_element = article_soup.find('article', class_='article-text').find('span', class_='rte')
            if byline_element:
                byline_text = byline_element.get_text(strip=True, separator='|').split('|')
                for part in byline_text:
                    if 'By ' in part:
                        authors = [a.strip() for a in part.replace('By ', '').split(' and ')]
                    if any(word in part.lower() for word in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                        date = part.strip()

            # Format title with metadata
            formatted_title = f"{title}"
            if authors or date:
                formatted_title += " ("
                if authors:
                    formatted_title += f"by:{','.join(authors)}"
                if date:
                    if authors: formatted_title += ";"
                    formatted_title += f"on:{date}"
                formatted_title += ")"

            # Extract main content
            content_parts = []
            main_articles = article_soup.find_all('article', class_='article-text')
            for article in main_articles:
                # Skip the byline article
                if 'By ' in article.get_text(): continue
                
                # Get all text elements including headings
                for element in article.find_all(['p', 'h2', 'h3', 'h4']):
                    text = element.get_text(strip=True, separator=' ')
                    if text:
                        content_parts.append(text)

            full_content = '\n'.join([t for t in content_parts if t.strip()])

            # Store in dictionary structure
            result[category_name][formatted_title] = full_content
        except Exception as e:
            print(f"Error retrieving {full_url}: {str(e)}")
            result[category_name].append('')
        finally:
            driver.close()
            driver.switch_to.window(original_window)
            time.sleep(1)  # Polite delay

# Save the results to a JSON file
with open('gartner_articles.json', 'w', encoding='utf-8') as f:
    print("dumping...")
    json.dump(result, f, ensure_ascii=False, indent=4)

driver.quit()
