#!/usr/bin/env python3
"""
Songs Analysis Script

Analyzes songs.json file focusing on song names, artists, and creation/modification dates.
Converts timestamps to readable ddmmyy format and extracts interesting insights.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class SongsAnalyzer:
    def __init__(self, json_file_path):
        """Initialize the analyzer with the songs JSON file."""
        self.json_file_path = json_file_path
        self.songs_data = None
        self.df = None
        
    def load_data(self):
        """Load and parse the songs JSON file."""
        print("Loading songs data...")
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.songs_data = data
        songs = data.get('songs', [])
        print(f"Loaded {len(songs)} songs")
        
        # Extract relevant fields and convert to DataFrame
        song_records = []
        for song in songs:
            record = {
                'id': song.get('id'),
                'name': song.get('name', ''),
                'singer': song.get('singer', ''),
                'dateCreated': song.get('dateCreated'),
                'dateModified': song.get('dateModified'),
                'categoryIds': song.get('categoryIds', [])
            }
            song_records.append(record)
        
        self.df = pd.DataFrame(song_records)
        print(f"Created DataFrame with {len(self.df)} records")
        
    def convert_timestamps(self):
        """Convert Unix timestamps to ddmmyy format."""
        print("Converting timestamps to ddmmyy format...")
        
        def timestamp_to_ddmmyy(timestamp):
            if pd.isna(timestamp) or timestamp == 0:
                return None
            try:
                dt = datetime.fromtimestamp(timestamp / 1000)  # Convert from milliseconds
                return dt.strftime('%d%m%y')
            except (ValueError, OSError):
                return None
        
        self.df['dateCreated_formatted'] = self.df['dateCreated'].apply(timestamp_to_ddmmyy)
        self.df['dateModified_formatted'] = self.df['dateModified'].apply(timestamp_to_ddmmyy)
        
        # Also create proper datetime columns for analysis
        self.df['dateCreated_dt'] = pd.to_datetime(self.df['dateCreated'], unit='ms', errors='coerce')
        self.df['dateModified_dt'] = pd.to_datetime(self.df['dateModified'], unit='ms', errors='coerce')
        
        print("Timestamp conversion completed")
        
    def analyze_artists(self):
        """Analyze artist-related insights."""
        print("\n=== ARTIST ANALYSIS ===")
        
        # Top artists by song count
        artist_counts = self.df['singer'].value_counts()
        print(f"\nTop 10 Most Prolific Artists:")
        for i, (artist, count) in enumerate(artist_counts.head(10).items(), 1):
            print(f"{i:2d}. {artist}: {count} songs")
        
        # Artist diversity
        total_artists = len(artist_counts)
        print(f"\nTotal unique artists: {total_artists}")
        print(f"Average songs per artist: {len(self.df) / total_artists:.1f}")
        
        # Artists with only one song vs multiple songs
        single_song_artists = len(artist_counts[artist_counts == 1])
        multi_song_artists = len(artist_counts[artist_counts > 1])
        print(f"Artists with only 1 song: {single_song_artists} ({single_song_artists/total_artists*100:.1f}%)")
        print(f"Artists with multiple songs: {multi_song_artists} ({multi_song_artists/total_artists*100:.1f}%)")
        
        return artist_counts
    
    def analyze_song_names(self):
        """Analyze song name patterns and insights."""
        print("\n=== SONG NAME ANALYSIS ===")
        
        # Language detection (basic Hebrew vs English)
        hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        self.df['has_hebrew'] = self.df['name'].apply(lambda x: bool(hebrew_pattern.search(str(x))))
        
        hebrew_songs = self.df['has_hebrew'].sum()
        english_songs = len(self.df) - hebrew_songs
        
        print(f"\nLanguage Distribution:")
        print(f"Songs with Hebrew characters: {hebrew_songs} ({hebrew_songs/len(self.df)*100:.1f}%)")
        print(f"Songs with only English characters: {english_songs} ({english_songs/len(self.df)*100:.1f}%)")
        
        # Song name length analysis
        self.df['name_length'] = self.df['name'].str.len()
        print(f"\nSong Name Length Statistics:")
        print(f"Average length: {self.df['name_length'].mean():.1f} characters")
        print(f"Shortest: {self.df['name_length'].min()} characters")
        print(f"Longest: {self.df['name_length'].max()} characters")
        
        # Most common words in song titles (English only for meaningful analysis)
        english_songs_df = self.df[~self.df['has_hebrew']]
        all_words = []
        for title in english_songs_df['name']:
            words = re.findall(r'\b[a-zA-Z]+\b', str(title).lower())
            all_words.extend(words)
        
        common_words = Counter(all_words).most_common(15)
        print(f"\nMost Common Words in English Song Titles:")
        for i, (word, count) in enumerate(common_words, 1):
            if len(word) > 2:  # Filter out very short words
                print(f"{i:2d}. '{word}': {count} times")
    
    def analyze_dates(self):
        """Analyze creation and modification date patterns."""
        print("\n=== DATE ANALYSIS ===")
        
        # Date range analysis
        min_created = self.df['dateCreated_dt'].min()
        max_created = self.df['dateCreated_dt'].max()
        min_modified = self.df['dateModified_dt'].min()
        max_modified = self.df['dateModified_dt'].max()
        
        print(f"\nDate Ranges:")
        print(f"Songs created between: {min_created.strftime('%d/%m/%Y')} and {max_created.strftime('%d/%m/%Y')}")
        print(f"Songs modified between: {min_modified.strftime('%d/%m/%Y')} and {max_modified.strftime('%d/%m/%Y')}")
        
        # Time gap between creation and modification
        self.df['days_to_modify'] = (self.df['dateModified_dt'] - self.df['dateCreated_dt']).dt.days
        
        # Filter out songs modified on the same day (likely just created)
        modified_later = self.df[self.df['days_to_modify'] > 0]
        
        if len(modified_later) > 0:
            print(f"\nModification Patterns:")
            print(f"Songs never modified: {len(self.df[self.df['days_to_modify'] == 0])} ({len(self.df[self.df['days_to_modify'] == 0])/len(self.df)*100:.1f}%)")
            print(f"Songs modified later: {len(modified_later)} ({len(modified_later)/len(self.df)*100:.1f}%)")
            print(f"Average days between creation and modification: {modified_later['days_to_modify'].mean():.1f}")
            print(f"Median days between creation and modification: {modified_later['days_to_modify'].median():.1f}")
        
        # Songs by year
        self.df['year_created'] = self.df['dateCreated_dt'].dt.year
        yearly_counts = self.df['year_created'].value_counts().sort_index()
        
        print(f"\nSongs Created by Year:")
        for year, count in yearly_counts.items():
            print(f"{year}: {count} songs")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("SONGS DATABASE SUMMARY REPORT")
        print("="*60)
        
        print(f"\nDatabase Overview:")
        print(f"Total songs: {len(self.df):,}")
        print(f"Unique artists: {self.df['singer'].nunique():,}")
        print(f"Date range: {self.df['dateCreated_dt'].min().strftime('%Y')} - {self.df['dateCreated_dt'].max().strftime('%Y')}")
        
        # Top insights
        top_artist = self.df['singer'].value_counts().iloc[0]
        top_artist_name = self.df['singer'].value_counts().index[0]
        
        print(f"\nKey Insights:")
        print(f"• Most prolific artist: {top_artist_name} ({top_artist} songs)")
        print(f"• Language mix: {self.df['has_hebrew'].sum()} Hebrew, {len(self.df) - self.df['has_hebrew'].sum()} English songs")
        print(f"• Most active year: {self.df['year_created'].value_counts().index[0]} ({self.df['year_created'].value_counts().iloc[0]} songs)")
        
        modified_songs = len(self.df[self.df['days_to_modify'] > 0])
        print(f"• Songs later modified: {modified_songs} ({modified_songs/len(self.df)*100:.1f}%)")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Songs Analysis...")
        print("="*50)
        
        self.load_data()
        self.convert_timestamps()
        
        # Run all analysis modules
        self.analyze_artists()
        self.analyze_song_names()
        self.analyze_dates()
        self.generate_summary_report()
        
        print(f"\nAnalysis completed! DataFrame saved as 'df' attribute with {len(self.df)} songs.")
        return self.df

def main():
    """Main function to run the analysis."""
    analyzer = SongsAnalyzer('songs.json')
    df = analyzer.run_full_analysis()
    
    # Return the analyzer object so it can be used in interactive mode
    return analyzer, df

if __name__ == "__main__":
    analyzer, df = main()
    print("\nTo access the data in interactive mode:")
    print("  analyzer = the SongsAnalyzer object with all methods")
    print("  df = the processed DataFrame with all songs data")
