import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class JupiterScraper:
    def __init__(self):
        self.base_url = "https://jupiter.money"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.scraped_data = []
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # or your preferred model
    
    def get_page_content(self, url):
        """Scrape content from a single page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            title = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            # Try different content selectors
            content_selectors = [
                'main', '.main-content', '#content', '.content',
                'article', '.article', '.post-content', 'section'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no main content found, get body text
            if not content:
                content = soup.get_text(separator=' ', strip=True)
            
            # Clean content
            content = self.clean_text(content)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'length': len(content)
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def clean_text(self, text):
        """Clean and normalize text"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Remove very short lines
        lines = [line.strip() for line in text.split('.') if len(line.strip()) > 20]
        
        return '. '.join(lines)
    
    def scrape_jupiter_pages(self):
        """Scrape key Jupiter Money pages"""
        # Comprehensive list of Jupiter Money pages
        pages_to_scrape = [
            # Core pages
            "https://jupiter.money/",
            "https://jupiter.money/savings-account/",
            "https://jupiter.money/pro-salary-account/",
            "https://jupiter.money/corporate-salary-account/",
            
            # Features
            "https://jupiter.money/pots/",
            "https://jupiter.money/payments/",
            "https://jupiter.money/bills-recharges/",
            "https://jupiter.money/pay-via-upi/",
            "https://jupiter.money/magic-spends/",
            
            # Credit Cards
            "https://jupiter.money/edge-plus-upi-rupay-credit-card/",
            "https://jupiter.money/edge-csb-rupay-credit-card/",
            "https://jupiter.money/edge-visa-credit-card/",
            
            # Loans
            "https://jupiter.money/loan/",
            "https://jupiter.money/loan-against-mutual-funds/",
            
            # Rewards & Investments
            "https://jupiter.money/rewards/",
            "https://jupiter.money/investments/",
            "https://jupiter.money/mutual-funds/",
            "https://jupiter.money/digi-gold/",
            "https://jupiter.money/flexi-fd/",
            "https://jupiter.money/recurring-deposits/",
            
            # Company
            "https://jupiter.money/money/",
            "https://jupiter.money/about-us/",
            "https://jupiter.money/careers/",
            "https://jupiter.money/contact-us/",
            
            # Community & Resources
            "https://jupiter.money/blog/",
            "https://jupiter.money/calculators/",
            "https://jupiter.money/calculators/home-loan-emi-calculator/",
            "https://jupiter.money/calculators/national-pension-scheme-calculator/",
            "https://jupiter.money/calculators/employee-provident-fund-calculator/",
            "https://jupiter.money/calculators/public-provident-fund-calculator/",
            "https://jupiter.money/calculators/salary-calculator/",
            "https://jupiter.money/calculators/personal-loan-emi-calculator/",
            "https://jupiter.money/calculators/credit-card-emi-calculator/",
        ]
        
        # Note: Skipping external links like Trello, Play Store, and subdomains
        # as they might have different structures or access restrictions
        
        print("Starting Jupiter Money comprehensive scraping...")
        print(f"📄 Total pages to scrape: {len(pages_to_scrape)}")
        
        successful_scrapes = 0
        failed_scrapes = 0
        
        for i, url in enumerate(pages_to_scrape, 1):
            print(f"Scraping [{i}/{len(pages_to_scrape)}]: {url}")
            
            page_data = self.get_page_content(url)
            if page_data and page_data['content']:
                self.scraped_data.append(page_data)
                successful_scrapes += 1
                print(f"✅ Success: {page_data['title'][:50]}... ({page_data['length']} chars)")
            else:
                failed_scrapes += 1
                print(f"❌ Failed: {url}")
            
            time.sleep(2)  # Be respectful - increased delay for more pages
        
        print(f"\n📊 Scraping Summary:")
        print(f"✅ Successful: {successful_scrapes}")
        print(f"❌ Failed: {failed_scrapes}")
        print(f"📄 Total content scraped: {sum(len(page['content']) for page in self.scraped_data)} characters")
        
        return self.scraped_data
    
    def save_data(self, filename="jupiter_data.json"):
        """Save scraped data to JSON file"""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Data saved to {filepath}")
        return filepath
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk)
            if len(chunk_text.strip()) > 50:  # Only add substantial chunks
                chunks.append(chunk_text.strip())
        
        return chunks
    
    def semantic_chunk_text(self, text, similarity_threshold=0.75, min_chunk_size=2, max_chunk_size=8):
        """
        Split text into semantically coherent chunks using sentence embeddings.
        Each chunk is a group of sentences that are semantically similar.
        """
        import re
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= min_chunk_size:
            return [' '.join(sentences)]

        # Compute embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        current_emb = [embeddings[0]]

        for i in range(1, len(sentences)):
            sim = np.dot(current_emb[-1], embeddings[i]) / (np.linalg.norm(current_emb[-1]) * np.linalg.norm(embeddings[i]) + 1e-8)
            # If similarity drops below threshold or chunk is too big, start new chunk
            if sim < similarity_threshold or len(current_chunk) >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_emb = [embeddings[i]]
            else:
                current_chunk.append(sentences[i])
                current_emb.append(embeddings[i])
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        # Filter out very small chunks
        return [chunk for chunk in chunks if len(chunk.split()) > 10]

    def prepare_data_for_embedding(self):
        """Prepare semantically chunked data for embedding"""
        prepared_data = []
        for page in self.scraped_data:
            chunks = self.semantic_chunk_text(page['content'])
            for i, chunk in enumerate(chunks):
                prepared_data.append({
                    'id': f"{page['url']}_{i}",
                    'url': page['url'],
                    'title': page['title'],
                    'content': chunk,
                    'chunk_id': i
                })
        return prepared_data

# Test the scraper
if __name__ == "__main__":
    scraper = JupiterScraper()
    
    # Scrape data
    data = scraper.scrape_jupiter_pages()
    
    # Save raw data
    scraper.save_data()
    
    # Prepare for embedding
    prepared_data = scraper.prepare_data_for_embedding()
    
    # Save prepared data
    with open('data/prepared_data.json', 'w', encoding='utf-8') as f:
        json.dump(prepared_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Comprehensive scraping complete!")
    print(f"📄 Pages scraped: {len(data)}")
    print(f"📝 Text chunks created: {len(prepared_data)}")
    print(f"💾 Data saved in 'data/' folder")
    print(f"📊 Total content: {sum(len(page['content']) for page in data)} characters")