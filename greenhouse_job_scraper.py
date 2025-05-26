"""
Greenhouse Job Scraper

This script scrapes job listings from Greenhouse job boards.
It extracts job titles, company names, locations, and job descriptions.
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

class GreenhouseJobScraper:
    """A scraper for Greenhouse job board listings."""
    
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the Greenhouse job scraper.
        
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
        Fetch a Greenhouse job page and return a BeautifulSoup object.
        
        Args:
            url: The URL to fetch
            
        Returns:
            BeautifulSoup object or None if request failed
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Save the HTML content to a file
            with open('greenhouse_response.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("HTML content saved to 'greenhouse_response.html'")
            
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page: {e}")
            return None
    
    def extract_job_listings(self, soup: BeautifulSoup, company_name: str = "") -> List[Dict[str, str]]:
        """
        Extract job listings from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object of the Greenhouse jobs page
            company_name: Name of the company (optional)
            
        Returns:
            List of dictionaries containing job information
        """
        job_listings = []
        
        # Find the job title (h1 tag)
        job_title_elem = soup.find('h1')
        if job_title_elem:
            job_title = job_title_elem.text.strip()
            
            # Find the location (often near the job title)
            location_elem = soup.find(string=lambda text: text and "Remote-" in text)
            location = location_elem.strip() if location_elem else "N/A"
            
            # Find the job description sections
            about_job_section = None
            job_description = ""
            
            # Look for "About the job" section
            for heading in soup.find_all(['h2', 'h3']):
                if "About the job" in heading.text:
                    about_job_section = heading.find_next('p')
                    if about_job_section:
                        job_description = about_job_section.text.strip()
                        break
            
            # If we couldn't find "About the job", try to get all content sections
            if not job_description:
                content_sections = soup.find_all(['p', 'li'])
                job_description = "\n".join([section.text.strip() for section in content_sections if section.text.strip()])
            
            # Extract job ID from URL
            job_id = url.split('/')[-1] if 'url' in locals() else "N/A"
            
            job_info = {
                'title': job_title,
                'company': company_name,
                'location': location,
                'description': job_description,
                'url': url if 'url' in locals() else "",
                'job_id': job_id
            }
            
            job_listings.append(job_info)
        
        return job_listings
    
    def scrape_job_details(self, job_url: str, company_name: str = "") -> Dict[str, str]:
        """
        Scrape detailed information about a specific job.
        
        Args:
            job_url: URL of the job listing
            company_name: Name of the company (optional)
            
        Returns:
            Dictionary containing detailed job information
        """
        soup = self.fetch_page(job_url)
        if not soup:
            return {'description': 'Failed to fetch job details'}
        
        # Extract job details using the same logic as extract_job_listings
        job_listings = self.extract_job_listings(soup, company_name)
        
        if job_listings:
            return job_listings[0]
        else:
            return {'description': 'No job details found'}
    
    def save_to_csv(self, job_listings: List[Dict[str, str]], filename: str = 'greenhouse_jobs.csv') -> None:
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
    
    def save_to_json(self, job_listings: List[Dict[str, str]], filename: str = 'greenhouse_jobs.json') -> None:
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
    """Main function to run the Greenhouse job scraper."""
    parser = argparse.ArgumentParser(description='Scrape Greenhouse job listings')
    parser.add_argument('--url', type=str, help='Greenhouse job URL to scrape', 
                        default='https://job-boards.greenhouse.io/remotecom/jobs/6531858003')
    parser.add_argument('--company', type=str, default='', help='Company name (optional)')
    parser.add_argument('--output', type=str, default='greenhouse_jobs.csv', help='Output filename')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv', help='Output format')
    args = parser.parse_args()
    
    url = args.url
    print(f"Scraping Greenhouse job from URL: {url}")
    
    # Extract company name from URL if not provided
    company_name = args.company
    if not company_name:
        # Try to extract company name from URL (e.g., remotecom from job-boards.greenhouse.io/remotecom)
        try:
            company_name = url.split('/')[3]
        except IndexError:
            company_name = "Unknown"
    
    scraper = GreenhouseJobScraper()
    soup = scraper.fetch_page(url)
    
    if soup:
        job_details = scraper.scrape_job_details(url, company_name)
        job_listings = [job_details] if job_details else []
        print(f"Found {len(job_listings)} job listings")
        
        # Save the results in the specified format
        if args.format == 'csv':
            scraper.save_to_csv(job_listings, args.output)
        else:
            scraper.save_to_json(job_listings, args.output.replace('.csv', '.json'))
    else:
        print("Failed to fetch the Greenhouse job page.")

if __name__ == "__main__":
    main()
