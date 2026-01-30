#!/usr/bin/env python3
"""
Generate Earth Engine Statistics with STAC Catalog Integration

This script processes Earth Engine usage statistics from Google Cloud Storage
and generates comprehensive JSON output with moving window analysis.
"""

import json
import logging
import os
import subprocess
import tempfile
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class STACCatalog:
    """Handle STAC catalog operations."""
    
    def __init__(self, catalog_url: str):
        """Initialize the STAC catalog.
        
        Args:
            catalog_url: URL to the STAC catalog JSON
        """
        self.catalog_url = catalog_url
        self.catalog_data = {}
        self.child_datasets = []  # List of child dataset info
        self.id_to_dataset = {}  # Map from asset ID to dataset info
        
    def fetch_catalog(self) -> bool:
        """Fetch the STAC catalog JSON and extract child datasets."""
        try:
            logger.info(f"Fetching STAC catalog from {self.catalog_url}")
            response = requests.get(self.catalog_url, timeout=30)
            response.raise_for_status()
            
            self.catalog_data = response.json()
            
            # Extract child links
            links = self.catalog_data.get('links', [])
            child_links = [link for link in links if link.get('rel') == 'child']
            
            logger.info(f"Found {len(child_links)} child datasets in STAC catalog")
            
            # Fetch each child dataset
            for child_link in child_links:
                try:
                    child_url = child_link.get('href')
                    child_title = child_link.get('title', '')
                    
                    if not child_url:
                        continue
                    
                    logger.debug(f"Fetching child dataset: {child_title}")
                    child_response = requests.get(child_url, timeout=30)
                    child_response.raise_for_status()
                    
                    child_data = child_response.json()
                    
                    # Extract the asset ID from the child data
                    # The ID in STAC typically looks like: "projects/neon-prod-earthengine/assets/CHM/001"
                    asset_id = child_data.get('id', '')
                    
                    dataset_info = {
                        'id': asset_id,
                        'title': child_data.get('title', child_title),
                        'description': child_data.get('description', ''),
                        'stac_url': child_url,
                        'type': child_data.get('type', ''),
                        'providers': child_data.get('providers', [])
                    }
                    
                    self.child_datasets.append(dataset_info)
                    
                    if asset_id:
                        self.id_to_dataset[asset_id] = dataset_info
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch child dataset {child_link.get('title', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(self.child_datasets)} STAC datasets")
            logger.info(f"Built lookup dictionary with {len(self.id_to_dataset)} unique asset IDs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch STAC catalog: {e}")
            return False
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset information by ID."""
        return self.id_to_dataset.get(dataset_id)
    
    def is_catalog_dataset(self, dataset_id: str) -> bool:
        """Check if a dataset ID is in the STAC catalog."""
        return dataset_id in self.id_to_dataset
    
    def get_catalog_dataset_ids(self) -> Set[str]:
        """Get all catalog dataset IDs."""
        return set(self.id_to_dataset.keys())

class EarthEngineStatsProcessor:
    """Process Earth Engine statistics with 30-day moving window data."""

    def __init__(self, stac_catalog_url: str):
        """Initialize the processor.
        
        Args:
            stac_catalog_url: URL to the STAC catalog JSON
        """
        self.bucket_name = "earthengine-stats"
        self.base_path = "providers/neon"
        self.start_date = date(2024, 9, 6) # Start date for moving window snapshots, first neon stats file was generated on 2024-09-06  
        self.moving_window_data = {}  # Store data by date
        self.processed_stats = {}
        self.stac_catalog = STACCatalog(stac_catalog_url)

    def _generate_date_range(self, end_date: Optional[date] = None) -> List[date]:
        """Generate list of dates for moving window snapshots."""
        if end_date is None:
            end_date = date.today() - timedelta(days=2)
        return [self.start_date + timedelta(days=x) for x in range((end_date - self.start_date).days + 1)]

    def _extract_dataset_id_from_name(self, dataset_name: str) -> str:
        """Extract clean dataset ID from dataset name by removing asset count suffix."""
        # Remove patterns like "/[123 assets]" from the end
        import re
        clean_name = re.sub(r'/\[\d+\s+assets\]$', '', dataset_name.strip())
        return clean_name

    def _download_csv_data(self, target_date: date) -> Optional[pd.DataFrame]:
        """Download and parse CSV data for a specific date (30-day window ending on this date)."""
        filename = f"earthengine_stats_{target_date.strftime('%Y-%m-%d')}.csv"
        gs_path = f"gs://{self.bucket_name}/{self.base_path}/{filename}"

        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
                temp_path = temp_file.name

            result = subprocess.run(
                ['gcloud', 'storage', 'cp', gs_path, temp_path],
                capture_output=True, text=True, timeout=30, check=False
            )

            if result.returncode != 0:
                if ("No such object" in result.stderr or 
                    "Not Found" in result.stderr or 
                    "may not exist" in result.stderr or
                    "HTTPError 403" in result.stderr):
                    logger.debug(f"File not found: {gs_path}")
                else:
                    logger.warning(f"Failed to download {gs_path}: {result.stderr}")
                return None

            df = pd.read_csv(temp_path)
            os.unlink(temp_path)

            if df.empty:
                logger.debug(f"Empty CSV file: {filename}")
                return None

            required_columns = ['Dataset', '30-day active users']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns in {filename}: {df.columns.tolist()}")
                return None

            return df

        except Exception as e:
            logger.debug(f"Error processing {filename}: {e}")
            return None

    def collect_all_data(self, end_date: Optional[date] = None, max_workers: int = 12) -> None:
        """Collect all moving window data from start_date to end_date."""
        logger.info("Fetching STAC catalog...")
        if not self.stac_catalog.fetch_catalog():
            logger.error("Failed to fetch STAC catalog. Continuing with empty catalog.")

        dates = self._generate_date_range(end_date)
        logger.info(f"Collecting data for {len(dates)} dates from {dates[0]} to {dates[-2]}")

        def download_and_process(target_date: date) -> tuple:
            """Download and process a single date."""
            df = self._download_csv_data(target_date)
            if df is not None:
                return (target_date, df)
            return (target_date, None)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_and_process, d): d for d in dates}
            
            completed = 0
            for future in as_completed(futures):
                target_date, df = future.result()
                if df is not None:
                    self.moving_window_data[target_date] = df
                
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(dates)} dates processed, {len(self.moving_window_data)} with data")

        logger.info(f"Data collection complete: {len(self.moving_window_data)} snapshots collected")

    def _calculate_moving_window_statistics(self) -> Dict:
        """Calculate comprehensive statistics from moving window data."""
        if not self.moving_window_data:
            logger.error("No moving window data available")
            return {}

        sorted_dates = sorted(self.moving_window_data.keys())
        logger.info(f"Processing {len(sorted_dates)} moving window snapshots")

        # Initialize data structures
        moving_window_snapshots = []
        all_dataset_stats = defaultdict(lambda: {'total_users': 0, 'appearances': 0, 'users_list': []})
        catalog_dataset_stats = defaultdict(lambda: {'total_users': 0, 'appearances': 0, 'users_list': []})

        # Process each snapshot
        for snapshot_date in sorted_dates:
            df = self.moving_window_data[snapshot_date]
            
            # Extract dataset statistics
            total_users = 0
            catalog_users = 0
            dataset_count = 0
            catalog_dataset_count = 0
            datasets = {}
            catalog_datasets = {}

            for _, row in df.iterrows():
                dataset_name = row['Dataset']
                users = int(row['30-day active users'])
                dataset_id = self._extract_dataset_id_from_name(dataset_name)

                total_users += users
                dataset_count += 1

                # Get STAC catalog info if available
                stac_info = self.stac_catalog.get_dataset_info(dataset_id)
                is_catalog = self.stac_catalog.is_catalog_dataset(dataset_id)

                # Store dataset info
                dataset_info = {
                    'users': users,
                    'title': stac_info.get('title', '') if stac_info else '',
                    'type': stac_info.get('type', '') if stac_info else ''
                }
                datasets[dataset_id] = dataset_info

                # Track all datasets
                all_dataset_stats[dataset_id]['total_users'] += users
                all_dataset_stats[dataset_id]['appearances'] += 1
                all_dataset_stats[dataset_id]['users_list'].append(users)

                # Track catalog datasets separately
                if is_catalog:
                    catalog_users += users
                    catalog_dataset_count += 1
                    catalog_datasets[dataset_id] = dataset_info
                    
                    catalog_dataset_stats[dataset_id]['total_users'] += users
                    catalog_dataset_stats[dataset_id]['appearances'] += 1
                    catalog_dataset_stats[dataset_id]['users_list'].append(users)

            # Create snapshot entry
            snapshot = {
                'date': snapshot_date.isoformat(),
                'total_users': total_users,
                'total_datasets': dataset_count,
                'catalog_total_users': catalog_users,
                'catalog_total_datasets': catalog_dataset_count,
                'datasets': datasets,
                'catalog_datasets': catalog_datasets
            }
            moving_window_snapshots.append(snapshot)

        # Calculate rankings
        dataset_rankings = {}
        for dataset_id, stats in all_dataset_stats.items():
            avg_users = stats['total_users'] / stats['appearances'] if stats['appearances'] > 0 else 0
            stac_info = self.stac_catalog.get_dataset_info(dataset_id)
            
            dataset_rankings[dataset_id] = {
                'avg_users': round(avg_users, 2),
                'total_users': stats['total_users'],
                'appearances': stats['appearances'],
                'title': stac_info.get('title', '') if stac_info else '',
                'type': stac_info.get('type', '') if stac_info else ''
            }

        catalog_dataset_rankings = {}
        for dataset_id, stats in catalog_dataset_stats.items():
            avg_users = stats['total_users'] / stats['appearances'] if stats['appearances'] > 0 else 0
            stac_info = self.stac_catalog.get_dataset_info(dataset_id)
            
            catalog_dataset_rankings[dataset_id] = {
                'avg_users': round(avg_users, 2),
                'total_users': stats['total_users'],
                'appearances': stats['appearances'],
                'title': stac_info.get('title', '') if stac_info else '',
                'type': stac_info.get('type', '') if stac_info else ''
            }

        # Calculate trends
        snapshot_trends = []
        catalog_snapshot_trends = []
        
        for i in range(1, len(moving_window_snapshots)):
            prev = moving_window_snapshots[i-1]
            curr = moving_window_snapshots[i]
            
            user_change = curr['total_users'] - prev['total_users']
            dataset_change = curr['total_datasets'] - prev['total_datasets']
            
            snapshot_trends.append({
                'date': curr['date'],
                'user_change': user_change,
                'dataset_change': dataset_change,
                'user_change_pct': round((user_change / prev['total_users'] * 100) if prev['total_users'] > 0 else 0, 2)
            })
            
            catalog_user_change = curr['catalog_total_users'] - prev['catalog_total_users']
            catalog_dataset_change = curr['catalog_total_datasets'] - prev['catalog_total_datasets']
            
            catalog_snapshot_trends.append({
                'date': curr['date'],
                'user_change': catalog_user_change,
                'dataset_change': catalog_dataset_change,
                'user_change_pct': round((catalog_user_change / prev['catalog_total_users'] * 100) if prev['catalog_total_users'] > 0 else 0, 2)
            })

        # Summary statistics
        latest_snapshot = moving_window_snapshots[-1]
        first_snapshot = moving_window_snapshots[0]
        
        total_datasets = latest_snapshot['total_datasets']
        latest_total_users = latest_snapshot['total_users']
        total_catalog_datasets = latest_snapshot['catalog_total_datasets']
        latest_catalog_total_users = latest_snapshot['catalog_total_users']
        
        # Calculate growth rates
        total_days = (sorted_dates[-1] - sorted_dates[0]).days
        if total_days > 0 and first_snapshot['total_users'] > 0:
            growth_rate = ((latest_total_users - first_snapshot['total_users']) / first_snapshot['total_users']) * 100
        else:
            growth_rate = 0
        
        if total_days > 0 and first_snapshot['catalog_total_users'] > 0:
            catalog_growth_rate = ((latest_catalog_total_users - first_snapshot['catalog_total_users']) / first_snapshot['catalog_total_users']) * 100
        else:
            catalog_growth_rate = 0

        # Find peaks
        peak_snapshot = max(moving_window_snapshots, key=lambda x: x['total_users'])
        peak_catalog_snapshot = max(moving_window_snapshots, key=lambda x: x['catalog_total_users'])

        # Get latest datasets for peak analysis
        latest_datasets = latest_snapshot['datasets']
        latest_catalog_datasets = latest_snapshot['catalog_datasets']

        # Peak dataset
        peak_dataset_info = {}
        if latest_datasets:
            peak_dataset = max(latest_datasets.items(), key=lambda x: x[1]['users'])
            peak_dataset_info = {
                'id': peak_dataset[0],
                'title': peak_dataset[1].get('title', ''),
                'total_users': peak_dataset[1]['users'],
                'avg_users': dataset_rankings.get(peak_dataset[0], {}).get('avg_users', 0),
                'type': peak_dataset[1].get('type', '')
            }

        # Peak catalog dataset
        if latest_catalog_datasets:
            peak_catalog_dataset = max(latest_catalog_datasets.items(), key=lambda x: x[1]['users'])
            peak_catalog_dataset_info = {
                'id': peak_catalog_dataset[0],
                'title': peak_catalog_dataset[1].get('title', ''),
                'total_users': peak_catalog_dataset[1]['users'],
                'avg_users': catalog_dataset_rankings.get(peak_catalog_dataset[0], {}).get('avg_users', 0),
                'type': peak_catalog_dataset[1].get('type', '')
            }
        else:
            peak_catalog_dataset_info = {'id': 'N/A', 'title': 'N/A', 'total_users': 0}

        stats = {
            "moving_window_snapshots": moving_window_snapshots,
            "data_type": "moving_window",
            "window_size_days": 30,
            "stac_catalog_info": {
                "total_catalog_datasets": len(self.stac_catalog.get_catalog_dataset_ids()),
                "catalog_url": self.stac_catalog.catalog_url,
                "catalog_id": self.stac_catalog.catalog_data.get('id', ''),
                "catalog_title": self.stac_catalog.catalog_data.get('title', ''),
                "last_updated": datetime.now().isoformat()
            },
            "summary": {
                "total_datasets": total_datasets,
                "total_usage_events": len(moving_window_snapshots),
                "latest_total_users": latest_total_users,
                "avg_users_per_dataset": round(latest_total_users / total_datasets, 2) if total_datasets > 0 else 0,
                "date_range": {
                    "start": sorted_dates[0].isoformat(),
                    "end": sorted_dates[-1].isoformat()
                },
                "growth_rate": round(growth_rate, 2)
            },
            "catalog_summary": {
                "total_catalog_datasets": total_catalog_datasets,
                "latest_catalog_total_users": latest_catalog_total_users,
                "catalog_percentage_of_datasets": round((total_catalog_datasets / total_datasets * 100) if total_datasets > 0 else 0, 2),
                "catalog_percentage_of_users": round((latest_catalog_total_users / latest_total_users * 100) if latest_total_users > 0 else 0, 2),
                "avg_users_per_catalog_dataset": round(latest_catalog_total_users / total_catalog_datasets, 2) if total_catalog_datasets > 0 else 0,
                "catalog_growth_rate": round(catalog_growth_rate, 2)
            },
            "peaks": {
                "peak_snapshot": peak_snapshot,
                "peak_dataset": peak_dataset_info,
                "peak_catalog_snapshot": peak_catalog_snapshot,
                "peak_catalog_dataset": peak_catalog_dataset_info
            },
            "dataset_rankings": dataset_rankings,
            "catalog_dataset_rankings": catalog_dataset_rankings,
            "snapshot_trends": snapshot_trends,
            "catalog_snapshot_trends": catalog_snapshot_trends
        }
        
        self.processed_stats = stats
        return stats

    def generate_json_output(self, output_file: str = "catalog_stats.json") -> None:
        """Generate JSON output file."""
        if not self.processed_stats:
            logger.error("No processed stats available. Run _calculate_moving_window_statistics first.")
            return

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully generated JSON output: {output_file}")
        except IOError as e:
            logger.error(f"Failed to write JSON output to {output_file}: {e}")

    def process_and_generate_json(self, end_date: Optional[date] = None,
                                  output_file: str = "catalog_stats.json",
                                  max_workers: int = 12) -> None:
        """Main method to process moving window data and generate JSON output."""
        logger.info("Starting Earth Engine moving window stats processing...")
        self.collect_all_data(end_date, max_workers)

        if not self.moving_window_data:
            logger.error("No moving window data collected. Aborting JSON generation.")
            return

        logger.info("Calculating moving window statistics...")
        self._calculate_moving_window_statistics()

        logger.info("Generating JSON output...")
        self.generate_json_output(output_file)

        # Log summary information
        logger.info(f"Processing complete. Summary:")
        logger.info(f"  - Moving window snapshots: {len(self.moving_window_data)}")
        logger.info(f"  - Latest snapshot datasets: {self.processed_stats['summary']['total_datasets']}")
        logger.info(f"  - STAC catalog datasets: {self.processed_stats['catalog_summary']['total_catalog_datasets']}")
        logger.info(f"  - Latest 30-day window users: {self.processed_stats['summary']['latest_total_users']:,}")
        logger.info(f"  - Catalog users: {self.processed_stats['catalog_summary']['latest_catalog_total_users']:,}")
        logger.info(f"  - Catalog percentage of users: {self.processed_stats['catalog_summary']['catalog_percentage_of_users']:.2f}%")
        logger.info(f"  - Snapshot date range: {self.processed_stats['summary']['date_range']['start']} to {self.processed_stats['summary']['date_range']['end']}")
        logger.info(f"  - Growth rate: {self.processed_stats['summary']['growth_rate']:.2f}%")
        logger.info(f"  - Catalog growth rate: {self.processed_stats['catalog_summary']['catalog_growth_rate']:.2f}%")

def main():
    """Main execution function."""
    STAC_CATALOG_URL = "https://storage.googleapis.com/earthengine-stac/catalog/neon-prod-earthengine/catalog.json"
    processor = EarthEngineStatsProcessor(stac_catalog_url=STAC_CATALOG_URL)
    processor.process_and_generate_json(end_date=None, output_file="catalog_stats.json", max_workers=12)

if __name__ == "__main__":
    main()
