"""
YouMed Medical Article Crawler
Crawls medical articles from YouMed.vn with multi-threaded support
"""

import requests
import json
import time
import random
import argparse
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _has_file_handler(log_path: Path) -> bool:
    """Return True if logger already has a file handler for log_path."""
    target = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if str(Path(handler.baseFilename).resolve()) == target:
                    return True
            except Exception:
                continue
    return False


class ListingPageParser:
    """Parses listing pages to extract article URLs and keywords"""
    
    def __init__(self, base_url: str = "https://youmed.vn"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://youmed.vn/'
        })
    
    def extract_article_urls(self, listing_url: str, delay: float = 1.0) -> List[Dict[str, str]]:
        """
        Extract all article URLs from a listing page (A-Z listing only)
        
        Args:
            listing_url: URL of the listing page
            delay: Delay between requests
            
        Returns:
            List of dicts: {"url": ..., "keyword": ...}
        """
        article_items = []
        logger.info(f"Extracting article URLs from: {listing_url}")
        
        try:
            time.sleep(delay + random.uniform(0, 0.5))
            
            response = self.session.get(listing_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # A-Z listing IDs: a-z-listing-letter-A-1 ... Z, and a-z-listing-letter-_-1
            letter_ids = [f"a-z-listing-letter-{chr(c)}-1" for c in range(ord('A'), ord('Z') + 1)]
            letter_ids.append("a-z-listing-letter-_-1")
            
            for letter_id in letter_ids:
                container = soup.find('div', id=letter_id)
                if not container:
                    continue
                
                for link in container.select('ul li a[href]'):
                    href = link.get('href', '').strip()
                    keyword = link.get_text(strip=True)
                    # keyword = str(link)
                    if not href:
                        continue
                    
                    full_url = urljoin(self.base_url, href)
                    if '/tin-tuc/' not in full_url:
                        continue
                    
                    article_items.append({
                        'url': full_url,
                        'keyword': keyword
                    })
            
        except requests.RequestException as e:
            logger.error(f"Error fetching listing {listing_url}: {e}")
        except Exception as e:
            logger.error(f"Error parsing listing {listing_url}: {e}")
        
        # Remove duplicates while preserving order (keep first keyword)
        seen = set()
        unique_items = []
        for item in article_items:
            if item['url'] not in seen:
                seen.add(item['url'])
                unique_items.append(item)
        
        logger.info(f"Total unique articles found: {len(unique_items)}")
        return unique_items


class ArticleParser:
    """Parses individual article pages to extract structured data"""
    
    def __init__(self, base_url: str = "https://youmed.vn"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://youmed.vn/'
        })
    
    def _extract_content_text(self, content_elem: BeautifulSoup) -> str:
        """
        Extract text from content element:
        - Remove inline tags (a, strong, etc.) without adding newlines
        - Keep h2/h3 tags as literal <h2>/<h3>
        - Preserve line breaks between block elements
        """
        # Remove TOC span markers inside headings if present
        for span in content_elem.select('span.ez-toc-section, span.ez-toc-section-end'):
            span.decompose()
        
        block_tags = {
            'p', 'div', 'section', 'article', 'ul', 'ol', 'li', 'br', 'table',
            'thead', 'tbody', 'tr', 'td', 'th', 'blockquote', 'figure', 'figcaption'
        }
        
        tokens = []
        
        def add_text(value: str):
            if value:
                tokens.append(value)
        
        def add_newline():
            tokens.append('\n')
        
        def walk(node):
            if isinstance(node, str):
                add_text(node)
                return
            
            name = getattr(node, 'name', None)
            if not name:
                return
            
            if name in {'h2', 'h3'}:
                text = node.get_text(strip=True)
                if text:
                    add_newline()
                    add_text(f"<{name}>{text}</{name}>")
                    add_newline()
                return
            
            if name == 'li':
                text = node.get_text(strip=True)
                if text:
                    add_newline()
                    add_text(f"<li>{text}</li>")
                    add_newline()
                return
            
            if name == 'figcaption':
                return
            
            if name in block_tags:
                add_newline()
                for child in node.children:
                    walk(child)
                add_newline()
                return
            
            # Inline/default: keep text without introducing newlines
            for child in node.children:
                walk(child)
        
        for child in content_elem.children:
            walk(child)
        
        text = ''.join(tokens)
        
        # Normalize whitespace and newlines
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if lines and lines[-1] == '' and line == '':
                continue
            lines.append(line)
        
        return '\n'.join(lines).strip()

    def parse_article(self, url: str, category: str = None, keyword: str = None) -> Optional[Dict[str, Any]]:
        """
        Parse an article page and extract structured data
        
        Returns:
            Dictionary with metadata, content, and references
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract metadata
            metadata = {
                'url': url,
                'category': category if category else 'unknown',
                'crawl_timestamp': datetime.now().isoformat(),
                'title': '',
                'author': '',
                'publish_date': '',
                'tags': [],
                'keyword': keyword or ''
            }
            
            # Extract title
            title_selectors = [
                'h1.entry-title',
                'h1.post-title',
                'h1.article-title',
                'h1',
                'title'
            ]
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    metadata['title'] = title_elem.get_text(strip=True)
                    break
            
            # Extract author - YouMed specific structure
            author_selectors = [
                'a.font-bold.text-primary',  # YouMed author link
                'a[href*="/bac-si/"]',  # Links to doctor profiles
                '.author',
                '.post-author',
                '.article-author',
                '[rel="author"]',
                'meta[property="article:author"]'
            ]
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    if author_elem.name == 'meta':
                        metadata['author'] = author_elem.get('content', '')
                    else:
                        metadata['author'] = author_elem.get_text(strip=True)
                    if metadata['author'] and metadata['author'] != 'Tác giả:':
                        break
            
            # Extract publish date - YouMed specific structure
            date_selectors = [
                'div.text-sm.text-black-700',  # YouMed date format
                'meta[property="article:published_time"]',
                'meta[name="publish-date"]',
                '.published',
                '.post-date',
                '.article-date',
                'time[datetime]',
                'time'
            ]
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    if date_elem.name == 'meta':
                        date_str = date_elem.get('content', '')
                    elif date_elem.get('datetime'):
                        date_str = date_elem.get('datetime')
                    else:
                        date_str = date_elem.get_text(strip=True)
                    
                    if date_str:
                        # Handle YouMed format: "Ngày Đăng: 20/11/2019 – Cập nhật lần cuối: 24/07/2021"
                        if 'Ngày Đăng:' in date_str or 'Ngày đăng:' in date_str:
                            # Extract the first date (publish date)
                            match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', date_str)
                            if match:
                                date_str = match.group(1)
                        
                        try:
                            # Try to parse the date
                            parsed_date = date_parser.parse(date_str, dayfirst=True)
                            metadata['publish_date'] = parsed_date.isoformat()
                        except:
                            metadata['publish_date'] = date_str
                        break
            
            # Extract tags
            tag_selectors = [
                '.tags a',
                '.post-tags a',
                '.article-tags a',
                'meta[name="keywords"]'
            ]
            for selector in tag_selectors:
                tag_elems = soup.select(selector)
                if tag_elems:
                    if tag_elems[0].name == 'meta':
                        keywords = tag_elems[0].get('content', '')
                        metadata['tags'] = [k.strip() for k in keywords.split(',') if k.strip()]
                    else:
                        metadata['tags'] = [tag.get_text(strip=True) for tag in tag_elems if tag.get_text(strip=True)]
                    if metadata['tags']:
                        break
            
            # Extract main content from the target container only
            content_text = ''
            content_elem = soup.select_one('div.prose.max-w-none.my-4.prose-a\\:text-primary')
            if content_elem:
                # Remove table of contents container
                toc = content_elem.find('div', id='ez-toc-container')
                if toc:
                    toc.decompose()
                
                # Remove scripts/styles just in case
                for unwanted in content_elem.select('script, style'):
                    unwanted.decompose()
                
                content_text = self._extract_content_text(content_elem)
            
            # Extract references
            references = []
            
            # Look for reference sections
            ref_selectors = [
                '.references',
                '.sources',
                '.tham-khao',
                '#references',
                '#sources'
            ]
            
            for selector in ref_selectors:
                ref_section = soup.select_one(selector)
                if ref_section:
                    ref_links = ref_section.find_all('a', href=True)
                    for link in ref_links:
                        ref_title = link.get_text(strip=True)
                        ref_url = urljoin(self.base_url, link.get('href', ''))
                        if ref_title and ref_url:
                            references.append({
                                'title': ref_title,
                                'url': ref_url
                            })
                    break
            
            # Also look for links in content that might be references
            if not references and content_elem:
                # Check for numbered references or citation patterns
                # Look for links that might be references (external links, numbered citations, etc.)
                all_links = content_elem.find_all('a', href=True)
                for link in all_links:
                    href = link.get('href', '')
                    # External links or links to other medical resources
                    if href.startswith('http') and 'youmed.vn' not in href:
                        ref_title = link.get_text(strip=True) or href
                        references.append({
                            'title': ref_title,
                            'url': href
                        })
            
            return {
                'metadata': metadata,
                'content': content_text,  # Clean text per custom rules
                'references': references
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching article {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing article {url}: {e}")
            return None


class YouMedCrawler:
    """Main crawler orchestrator with multi-threaded support"""
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        workers: int = 5,
        delay: float = 1.0,
        max_retries: int = 3,
        test_mode: bool = False,
        limit: Optional[int] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.workers = workers
        self.delay = delay
        self.max_retries = max_retries
        self.test_mode = test_mode
        self.limit = limit if limit else (2 if test_mode else None)
        
        self.listing_parser = ListingPageParser()
        self.article_parser = ArticleParser()
        
        # Thread-safe counters and locks
        self.write_lock = Lock()
        self.stats_lock = Lock()
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0
        }
        
        # Output files
        self.output_files_by_category = {
            'disease': self.output_dir / "youmed_articles_disease.jsonl",
            'drug': self.output_dir / "youmed_articles_drug.jsonl",
            'medicine': self.output_dir / "youmed_articles_medicine.jsonl",
            'body-part': self.output_dir / "youmed_articles_body-part.jsonl",
            'other': self.output_dir / "youmed_articles_other.jsonl"
        }
        self.log_file = self.output_dir / "crawler.log"
        self.failed_urls_file = self.output_dir / "failed_urls.txt"
        
        # Setup file logging
        if not _has_file_handler(self.log_file):
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        
        logger.info(f"YouMedCrawler initialized: workers={workers}, delay={delay}, test_mode={test_mode}, limit={limit}")
    
    def get_article_urls_from_listings(self, listing_urls: List[str]) -> List[Dict[str, str]]:
        """Extract all article URLs from listing pages with their category and keyword"""
        all_items = []
        
        # Map listing URLs to categories
        category_map = {
            'https://youmed.vn/tin-tuc/trieu-chung-benh/': 'disease',
            'https://youmed.vn/tin-tuc/duoc/': 'drug',
            'https://youmed.vn/tin-tuc/y-hoc-co-truyen/duoc-lieu/': 'medicine',
            'https://youmed.vn/tin-tuc/hieu-ve-co-the-ban/': 'body-part'
        }
        
        for listing_url in listing_urls:
            try:
                items = self.listing_parser.extract_article_urls(listing_url, delay=self.delay)
                category = category_map.get(listing_url, 'other')
                for item in items:
                    all_items.append({
                        'url': item['url'],
                        'keyword': item.get('keyword', ''),
                        'category': category
                    })
            except Exception as e:
                logger.error(f"Error extracting URLs from {listing_url}: {e}")
        
        # Remove duplicates (keep first category/keyword seen)
        seen = set()
        unique_items = []
        for item in all_items:
            if item['url'] not in seen:
                seen.add(item['url'])
                unique_items.append(item)
        
        # Apply limit if in test mode
        if self.limit:
            unique_items = unique_items[:self.limit]
            logger.info(f"Limited to {self.limit} articles for testing")
        
        return unique_items
    
    def crawl_article(self, url: str, category: str = None, keyword: str = None) -> Optional[Dict[str, Any]]:
        """Crawl a single article with retry logic"""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay + random.uniform(0, 0.3))
                article_data = self.article_parser.parse_article(url, category=category, keyword=keyword)
                
                if article_data:
                    with self.stats_lock:
                        self.stats['success'] += 1
                    return article_data
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        with self.stats_lock:
                            self.stats['failed'] += 1
                        return None
                        
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    with self.stats_lock:
                        self.stats['failed'] += 1
                    return None
        
        return None
    
    def save_article(self, article_data: Dict[str, Any]):
        """Thread-safe JSONL writing by category"""
        category = article_data.get('metadata', {}).get('category', 'other')
        output_file = self.output_files_by_category.get(category, self.output_files_by_category['other'])
        with self.write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False)
                f.write('\n')
    
    def log_failed_url(self, url: str):
        """Log failed URL to file"""
        with self.write_lock:
            with open(self.failed_urls_file, 'a', encoding='utf-8') as f:
                f.write(f"{url}\n")
    
    def crawl(self, listing_urls: List[str]):
        """Main crawl method with multi-threading"""
        logger.info("=" * 60)
        logger.info("Starting YouMed crawler")
        logger.info("=" * 60)
        
        # Phase 1: Extract article URLs from listing pages
        logger.info("Phase 1: Extracting article URLs from listing pages...")
        article_items = self.get_article_urls_from_listings(listing_urls)
        
        if not article_items:
            logger.warning("No article URLs found!")
            return
        
        self.stats['total'] = len(article_items)
        logger.info(f"Found {len(article_items)} articles to crawl")
        
        # Clear output files
        for output_file in self.output_files_by_category.values():
            if output_file.exists():
                output_file.unlink()
        if self.failed_urls_file.exists():
            self.failed_urls_file.unlink()
        
        # Phase 2: Crawl articles with multi-threading
        logger.info("Phase 2: Crawling articles with multi-threading...")
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.crawl_article, item['url'], item['category'], item.get('keyword', '')): item
                for item in article_items
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(article_items), desc="Crawling articles") as pbar:
                for future in as_completed(future_to_url):
                    item = future_to_url[future]
                    try:
                        article_data = future.result()
                        if article_data:
                            self.save_article(article_data)
                        else:
                            self.log_failed_url(item['url'])
                    except Exception as e:
                        logger.error(f"Error processing {item['url']}: {e}")
                        self.log_failed_url(item['url'])
                        with self.stats_lock:
                            self.stats['failed'] += 1
                    
                    pbar.update(1)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Crawling completed!")
        logger.info(f"Total articles: {self.stats['total']}")
        logger.info(f"Successfully crawled: {self.stats['success']}")
        logger.info(f"Failed: {self.stats['failed']}")
        non_empty_outputs = []
        for output_file in self.output_files_by_category.values():
            if output_file.exists() and output_file.stat().st_size > 0:
                non_empty_outputs.append(str(output_file))
        logger.info(f"Output files ({len(non_empty_outputs)} non-empty): {', '.join(non_empty_outputs) if non_empty_outputs else 'none'}")
        logger.info(f"Failed URLs: {self.failed_urls_file}")
        logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='YouMed Medical Article Crawler')
    parser.add_argument('--test', action='store_true', help='Test mode (crawl only 2 articles)')
    parser.add_argument('--limit', type=int, help='Limit number of articles to crawl')
    parser.add_argument('--workers', type=int, default=5, help='Number of worker threads (default: 5)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Output directory (default: data/raw)')
    parser.add_argument('--categories', type=str, help='Comma-separated list of categories to crawl')
    
    args = parser.parse_args()
    
    # Default listing URLs
    default_listings = [
        "https://youmed.vn/tin-tuc/trieu-chung-benh/",
        "https://youmed.vn/tin-tuc/duoc/",
        "https://youmed.vn/tin-tuc/y-hoc-co-truyen/duoc-lieu/",
        "https://youmed.vn/tin-tuc/hieu-ve-co-the-ban/"
    ]
    
    # Filter categories if specified
    if args.categories:
        category_list = [c.strip() for c in args.categories.split(',')]
        listing_urls = [
            url for url in default_listings 
            if any(cat in url for cat in category_list)
        ]
        if not listing_urls:
            logger.warning(f"No matching categories found for: {args.categories}")
            listing_urls = default_listings
    else:
        listing_urls = default_listings
    
    # Determine limit
    limit = None
    if args.test:
        limit = 2
    elif args.limit:
        limit = args.limit
    
    # Create crawler and run
    crawler = YouMedCrawler(
        output_dir=args.output_dir,
        workers=args.workers,
        delay=args.delay,
        test_mode=args.test,
        limit=limit
    )
    
    crawler.crawl(listing_urls)


if __name__ == "__main__":
    main()
