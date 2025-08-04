import requests
from bs4 import BeautifulSoup
import json
import time
import re
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
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def extract_contact_info(self, soup, url, raw_text):
        """Special extraction for contact pages"""
        contact_data = {
            'phones': [],
            'emails': [],
            'chat_info': '',
            'hours': '',
            'address': ''
        }
        
        # Extract phone numbers with various patterns
        phone_patterns = [
            r'\+91[\s-]?\d{5}[\s-]?\d{5}',  # +91 86550 55086
            r'\d{3}-\d{8}',                 # 080-44353535
            r'1800[\s-]?\d{3}[\s-]?\d{4}',  # Toll-free
            r'\b\d{10}\b'                   # 10-digit numbers
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, raw_text)
            contact_data['phones'].extend(phones)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@jupiter\.money\b'
        emails = re.findall(email_pattern, raw_text)
        contact_data['emails'].extend(emails)
        
        # Extract hours information
        hours_patterns = [
            r'(\d{1,2}\s*(?:am|pm)\s*(?:to|-)\s*\d{1,2}\s*(?:am|pm))',
            r'(weekdays?(?:\s+from)?\s+\d{1,2}\s*(?:am|pm))',
            r'(monday\s+to\s+\w+.*?\d{1,2}\s*(?:am|pm))'
        ]
        
        for pattern in hours_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            if matches:
                contact_data['hours'] = matches[0]
                break
        
        # Extract chat info
        if 'chat' in raw_text.lower():
            chat_sentences = [s.strip() for s in raw_text.split('.') 
                            if 'chat' in s.lower() and len(s.strip()) > 10]
            contact_data['chat_info'] = '. '.join(chat_sentences[:2])
        
        return contact_data
    
    def get_page_content(self, url):
        """Enhanced content extraction with special handling for contact pages"""
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
            
            # Get raw text for contact info extraction
            raw_text = soup.get_text(separator=' ', strip=True)
            
            # Special handling for contact pages
            is_contact_page = 'contact' in url.lower() or 'contact' in title.lower()
            
            if is_contact_page:
                contact_info = self.extract_contact_info(soup, url, raw_text)
                # Create structured contact content
                content = self.create_structured_contact_content(contact_info, raw_text)
            else:
                # Regular content extraction
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
                
                if not content:
                    content = raw_text
                
                content = self.clean_text(content)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'length': len(content),
                'is_contact_page': is_contact_page
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def create_structured_contact_content(self, contact_info, raw_text):
        """Create well-structured contact content"""
        structured_parts = []
        
        # Phone section
        if contact_info['phones']:
            phones_text = f"Jupiter Money Customer Support Phone Numbers: {', '.join(set(contact_info['phones']))}. "
            phones_text += "Call customer service helpline for assistance. Live agent support phone number. "
            if contact_info['hours']:
                phones_text += f"Available {contact_info['hours']}. "
            structured_parts.append(phones_text)
        
        # Email section  
        if contact_info['emails']:
            emails_text = f"Jupiter Money Customer Support Email: {', '.join(set(contact_info['emails']))}. "
            emails_text += "Send email to customer service for help and inquiries. Customer support email address. "
            structured_parts.append(emails_text)
        
        # Chat section
        if contact_info['chat_info']:
            chat_text = f"Jupiter Money Chat Support: {contact_info['chat_info']}. "
            chat_text += "In-app chat customer service. Live chat support through Jupiter app. "
            structured_parts.append(chat_text)
        
        # Hours section (if not already included)
        if contact_info['hours'] and not any('hours' in part.lower() for part in structured_parts):
            hours_text = f"Jupiter Customer Support Hours: {contact_info['hours']}. "
            hours_text += "Customer service availability schedule. Support timing. "
            structured_parts.append(hours_text)
        
        # Add cleaned original content for additional context
        cleaned_original = self.clean_text(raw_text)
        if cleaned_original and len(cleaned_original) > 100:
            structured_parts.append(cleaned_original)
        
        return ' '.join(structured_parts)
    
    def clean_text(self, text):
        """Enhanced text cleaning that preserves important contact info"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # For contact pages, be more gentle with cleaning
        # Keep phone numbers, emails, and important punctuation
        text = re.sub(r'[^\w\s.,!?@+-]', '', text)
        
        # Remove very short fragments but preserve contact info
        lines = []
        for line in text.split('.'):
            line = line.strip()
            # Keep lines with contact info or substantial content
            if (len(line) > 15 or 
                re.search(r'\d{3}-\d{8}', line) or  # Phone patterns
                re.search(r'\+91', line) or
                '@jupiter.money' in line or
                any(word in line.lower() for word in ['support', 'contact', 'email', 'phone', 'chat'])):
                lines.append(line)
        
        return '. '.join(lines)
    
    def scrape_jupiter_pages(self):
        """Scrape key Jupiter Money pages with enhanced contact handling"""
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
            
            # Company - CONTACT PAGE PRIORITIZED
            "https://jupiter.money/contact-us/",  # Process this early
            "https://jupiter.money/money/",
            "https://jupiter.money/about-us/",
            "https://jupiter.money/careers/",
            
            # Community & Resources
            "https://jupiter.money/blog/",
            "https://jupiter.money/calculators/",
        ]
        
        print("Starting Jupiter Money enhanced scraping...")
        print(f"📄 Total pages to scrape: {len(pages_to_scrape)}")
        
        successful_scrapes = 0
        failed_scrapes = 0
        
        for i, url in enumerate(pages_to_scrape, 1):
            print(f"Scraping [{i}/{len(pages_to_scrape)}]: {url}")
            
            page_data = self.get_page_content(url)
            if page_data and page_data['content']:
                self.scraped_data.append(page_data)
                successful_scrapes += 1
                
                # Special logging for contact pages
                if page_data.get('is_contact_page'):
                    print(f"🎯 CONTACT PAGE: {page_data['title'][:50]}... ({page_data['length']} chars)")
                    # Preview first 200 chars of contact content
                    preview = page_data['content'][:200] + "..."
                    print(f"   Preview: {preview}")
                else:
                    print(f"✅ Success: {page_data['title'][:50]}... ({page_data['length']} chars)")
            else:
                failed_scrapes += 1
                print(f"❌ Failed: {url}")
            
            time.sleep(1)  # Reasonable delay
        
        print(f"\n📊 Scraping Summary:")
        print(f"✅ Successful: {successful_scrapes}")
        print(f"❌ Failed: {failed_scrapes}")
        
        # Count contact pages
        contact_pages = sum(1 for page in self.scraped_data if page.get('is_contact_page'))
        print(f"📞 Contact pages found: {contact_pages}")
        
        return self.scraped_data
    
    def semantic_chunk_text(self, text, similarity_threshold=0.75, min_chunk_size=2, max_chunk_size=8):
        """Enhanced semantic chunking with contact-aware processing"""
        import re
        
        # For contact content, use smaller chunks to preserve specific info
        if any(keyword in text.lower() for keyword in ['phone', 'email', 'contact', 'support']):
            # Use sentence-based chunking for contact info
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Group contact-related sentences
            contact_chunks = []
            current_chunk = []
            
            for sentence in sentences:
                current_chunk.append(sentence)
                
                # Create chunk when we have enough sentences or hit a topic boundary
                if (len(current_chunk) >= 3 or 
                    any(keyword in sentence.lower() for keyword in ['phone', 'email', 'chat', 'hours'])):
                    
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) > 5:  # Minimum chunk size
                        contact_chunks.append(chunk_text)
                    current_chunk = []
            
            # Add remaining sentences
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) > 5:
                    contact_chunks.append(chunk_text)
            
            return contact_chunks if contact_chunks else [text]
        
        # Regular semantic chunking for non-contact content
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= min_chunk_size:
            return [' '.join(sentences)]

        embeddings = self.embedding_model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        current_emb = [embeddings[0]]

        for i in range(1, len(sentences)):
            sim = np.dot(current_emb[-1], embeddings[i]) / (
                np.linalg.norm(current_emb[-1]) * np.linalg.norm(embeddings[i]) + 1e-8
            )
            
            if sim < similarity_threshold or len(current_chunk) >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_emb = [embeddings[i]]
            else:
                current_chunk.append(sentences[i])
                current_emb.append(embeddings[i])
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk for chunk in chunks if len(chunk.split()) > 10]
    
    def prepare_data_for_embedding(self):
        """Enhanced data preparation with contact page priority"""
        prepared_data = []
        
        for page in self.scraped_data:
            chunks = self.semantic_chunk_text(page['content'])
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': f"{page['url']}_{i}",
                    'url': page['url'],
                    'title': page['title'],
                    'content': chunk,
                    'chunk_id': i,
                    'is_contact': page.get('is_contact_page', False)
                }
                prepared_data.append(chunk_data)
        
        # Sort to prioritize contact chunks
        prepared_data.sort(key=lambda x: (not x['is_contact'], x['url'], x['chunk_id']))
        
        return prepared_data
    
    def save_data(self, filename="jupiter_data.json"):
        """Save scraped data to JSON file"""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Data saved to {filepath}")
        return filepath

# Test the enhanced scraper
if __name__ == "__main__":
    scraper = JupiterScraper()
    
    # Scrape data with enhanced contact handling
    data = scraper.scrape_jupiter_pages()
    
    # Save raw data
    scraper.save_data("enhanced_jupiter_data.json")
    
    # Prepare for embedding with contact priority
    prepared_data = scraper.prepare_data_for_embedding()
    
    # Save prepared data
    with open('data/enhanced_prepared_data.json', 'w', encoding='utf-8') as f:
        json.dump(prepared_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Enhanced scraping complete!")
    print(f"📄 Pages scraped: {len(data)}")
    print(f"📝 Text chunks created: {len(prepared_data)}")
    
    # Show contact chunks specifically
    contact_chunks = [chunk for chunk in prepared_data if chunk['is_contact']]
    print(f"📞 Contact chunks: {len(contact_chunks)}")
    
    if contact_chunks:
        print("\n📞 Contact Content Preview:")
        for chunk in contact_chunks[:3]:  # Show first 3 contact chunks
            print(f"   {chunk['content'][:150]}...")
    
    print(f"💾 Enhanced data saved in 'data/' folder")