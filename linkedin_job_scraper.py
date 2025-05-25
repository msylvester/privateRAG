"""
LinkedIn Job Scraper

This script scrapes job listings from LinkedIn for AI Film Maker positions.
It extracts job titles, company names, locations, and posting dates.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import json
from typing import List, Dict, Any, Optional
import argparse

class LinkedInJobScraper:
    """A scraper for LinkedIn job listings."""
    
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the LinkedIn job scraper.
        
        Args:
            headers: Custom headers for HTTP requests (optional)
        """
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a LinkedIn page and return a BeautifulSoup object.
        
        Args:
            url: The URL to fetch
            
        Returns:
            BeautifulSoup object or None if request failed
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page: {e}")
            return None
    
    def extract_job_listings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract job listings from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object of the LinkedIn jobs page
            
        Returns:
            List of dictionaries containing job information
        """
        job_listings = []
        
        # Find all job cards
        job_cards = soup.find_all('div', class_='job-search-card')
        
        if not job_cards:
            # If the expected structure isn't found, try an alternative approach
            # This is a simplified example based on the provided HTML structure
            job_items = soup.find_all('li')
            
            for item in job_items:
                job_link = item.find('a', href=True)
                if not job_link:
                    continue
                
                job_title_elem = job_link.find('h3')
                company_elem = item.find('h4')
                location_elem = item.find('span', string=lambda text: text and "United States" in text)
                
                if job_title_elem and company_elem:
                    job_info = {
                        'title': job_title_elem.text.strip(),
                        'company': company_elem.text.strip(),
                        'location': location_elem.text.strip() if location_elem else "N/A",
                        'url': job_link.get('href', ''),
                        'job_id': job_link.get('href', '').split('/')[-1].split('?')[0] if job_link.get('href') else "N/A"
                    }
                    job_listings.append(job_info)
        
        return job_listings
    
    def scrape_job_details(self, job_url: str) -> Dict[str, str]:
        """
        Scrape detailed information about a specific job.
        
        Args:
            job_url: URL of the job listing
            
        Returns:
            Dictionary containing detailed job information
        """
        soup = self.fetch_page(job_url)
        if not soup:
            return {'description': 'Failed to fetch job details'}
        
        # Extract job description
        description_elem = soup.find('div', class_='description__text')
        description = description_elem.text.strip() if description_elem else "No description available"
        
        # Extract other details as needed
        return {
            'description': description,
            # Add more fields as needed
        }
    
    def save_to_csv(self, job_listings: List[Dict[str, str]], filename: str = 'linkedin_jobs.csv') -> None:
        """
        Save job listings to a CSV file.
        
        Args:
            job_listings: List of job listing dictionaries
            filename: Output CSV filename
        """
        if not job_listings:
            print("No job listings to save.")
            return
        
        df = pd.DataFrame(job_listings)
        df.to_csv(filename, index=False)
        print(f"Saved {len(job_listings)} job listings to {filename}")
    
    def save_to_json(self, job_listings: List[Dict[str, str]], filename: str = 'linkedin_jobs.json') -> None:
        """
        Save job listings to a JSON file.
        
        Args:
            job_listings: List of job listing dictionaries
            filename: Output JSON filename
        """
        if not job_listings:
            print("No job listings to save.")
            return
        
        with open(filename, 'w') as f:
            json.dump(job_listings, f, indent=2)
        print(f"Saved {len(job_listings)} job listings to {filename}")

def main():
    """Main function to run the LinkedIn job scraper."""
    parser = argparse.ArgumentParser(description='Scrape LinkedIn job listings')
    parser.add_argument('--keywords', type=str, default='ai film maker', help='Job search keywords')
    parser.add_argument('--location', type=str, default='United States', help='Job location')
    parser.add_argument('--output', type=str, default='linkedin_jobs.csv', help='Output filename')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv', help='Output format')
    args = parser.parse_args()
    
    # Format the URL with the search parameters
    keywords = args.keywords.replace(' ', '%20')
    location_id = '103644278'  # This is the geoId for United States
    url = f"https://www.linkedin.com/jobs/search/?keywords={keywords}&geoId={location_id}"
    
    print(f"Scraping LinkedIn jobs for '{args.keywords}' in {args.location}...")
    
    scraper = LinkedInJobScraper()
    soup = scraper.fetch_page(url)
    
    if soup:
        job_listings = scraper.extract_job_listings(soup)
        print(f"Found {len(job_listings)} job listings")
        
        # Save the results in the specified format
        if args.format == 'csv':
            scraper.save_to_csv(job_listings, args.output)
        else:
            scraper.save_to_json(job_listings, args.output.replace('.csv', '.json'))
    else:
        print("Failed to fetch the LinkedIn jobs page.")
        print("Note: LinkedIn may block automated scraping. Consider using their official API.")

if __name__ == "__main__":
    main()
