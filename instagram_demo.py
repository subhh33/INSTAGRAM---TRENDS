"""
instagram_demo.py
===================

This script demonstrates a simple Instagram performance analysis and
content‑quality control (QC) bot.  It loads a sample dataset of
Instagram posts (stored in `instagram_sample_data.csv`), computes
engagement metrics, identifies trending hashtags, evaluates posting
times, and runs basic QC checks on the captions.  The goal is to
illustrate how data and simple heuristics can be combined to produce
actionable insights for a social‑media team.

How to run
----------
Run the script with a Python interpreter (Python 3.7+):

    python instagram_demo.py

The output will include:
  * A table of the original posts with calculated engagement rates.
  * A list of the top posts sorted by engagement rate.
  * A ranked list of hashtags by frequency.
  * Average engagement rates by posting hour (helps identify optimal
    posting times).
  * QC recommendations for each caption, highlighting areas such as
    caption length, sentence length, hashtag usage, and whether
    trending/niche hashtags are used.

The dataset provided in `instagram_sample_data.csv` is entirely
synthetic; you can replace it with your own CSV file containing
similar columns (Post, Likes, Comments, Followers, Hashtags,
Timestamp) to analyze real data.

"""

import pandas as pd
import re
from collections import Counter
import itertools


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Instagram data from a CSV file.

    The CSV is expected to have the following columns:
    - Post: caption text including hashtags.
    - Likes: integer count of likes.
    - Comments: integer count of comments.
    - Followers: integer count of followers at the time of posting.
    - Hashtags: comma‑separated list of hashtags (without the `#` symbol).
    - Timestamp: ISO‑8601 datetime string (YYYY‑MM‑DD HH:MM:SS).

    Returns a DataFrame with a parsed Timestamp column.
    """
    df = pd.read_csv(filepath)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # Convert comma‑separated hashtags to lists
    df["Hashtags"] = df["Hashtags"].apply(lambda x: [h.strip() for h in str(x).split(",") if h.strip()])
    return df


def compute_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement rate for each post.

    Engagement rate is defined as (Likes + Comments) / Followers.
    Adds a new column 'EngagementRate' to the DataFrame.
    """
    df = df.copy()
    df["EngagementRate"] = (df["Likes"] + df["Comments"]) / df["Followers"]
    return df


def analyze_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    """Count the frequency of each hashtag across all posts.

    Returns a DataFrame with columns 'Hashtag' and 'Count', sorted
    descending by count.
    """
    all_tags = list(itertools.chain.from_iterable(df["Hashtags"]))
    counts = Counter(all_tags)
    hashtag_df = pd.DataFrame(counts.items(), columns=["Hashtag", "Count"])
    return hashtag_df.sort_values(by="Count", ascending=False)


def analyze_posting_times(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average engagement by hour of day.

    Extracts the hour from the Timestamp and computes the mean
    engagement rate for each hour.
    """
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    hourly = df.groupby("Hour")["EngagementRate"].mean().reset_index()
    return hourly.sort_values(by="EngagementRate", ascending=False)


def qc_check(df: pd.DataFrame) -> pd.DataFrame:
    """Run basic content quality checks on each caption.

    The checks include:
    - Word count and sentence count.
    - Average word and sentence lengths.
    - Hashtag count per caption.
    - Whether any trending/niche hashtags are present.
    - Suggestions for improvement (e.g., shorter captions, breaking
      long sentences, using trending hashtags, or adjusting the number
      of hashtags).

    Returns a DataFrame with these metrics and suggestions alongside
    the original caption.
    """
    trending = {
        # Art‑related tags (extract from Metricool article)
        'art', 'artist', 'artwork', 'nailart', 'digitalart', 'arte', 'instaart', 'artistsoninstagram',
        'streetart', 'artoftheday', 'contemporaryart', 'fanart', 'artofvisuals', 'artsy', 'abstractart',
        'fineart', 'artgallery', 'instaartist', 'arts', 'myart', 'artistic', 'modernart', 'animeart',
        'picsart', 'tattooart', 'nailsart', 'artists', 'urbanart', 'artistsofinstagram',
        # Pet‑related tags
        'pet', 'pets', 'petstagram', 'petsofinstagram', 'instapet', 'petsagram', 'petlovers',
        'instapets', 'petshop', 'mypet', 'petphotography', 'petlove', 'cutepets', 'petfriendly',
        'petscorner', 'petportrait', 'ilovemypet', 'happy_pet', 'petsofinsta', 'picpets', 'lovepets',
        'nationalpetday', 'worldofcutepets', 'petsmart', 'petsitting', 'petlife', 'petsgram', 'petgrooming',
        'petofinstagram',
        # Travel‑related tags
        'travel', 'travelgram', 'travelphotography', 'instatravel', 'traveling', 'travelling', 'travelblogger',
        'traveler', 'traveller', 'traveltheworld', 'igtravel', 'travelingram', 'travelblog', 'mytravelgram',
        'travels', 'instatraveling', 'traveladdict', 'travelphoto', 'traveldiaries', 'travelawesome',
        'wanderlust', 'wanderlusting', 'wanderluster', 'wanderlusters', 'wanderlustwednesday', 'visualwanderlust',
        'wanderlustlife', 'asianwanderlust', 'wanderlustvibes', 'travellife'
    }
    results = []
    for _, row in df.iterrows():
        caption = row["Post"]
        hashtags = [h.lower() for h in row["Hashtags"]]
        text_only = re.sub(r"#\w+", "", caption)  # remove hashtags
        text_only = re.sub(r"[^\w\s\.\!\?]", "", text_only)  # strip emojis/special chars
        words = re.findall(r"\b\w+\b", text_only)
        sentences = [s.strip() for s in re.split(r"[\.!?]+", text_only) if s.strip()]
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
        avg_sent_len = word_count / sentence_count if sentence_count else 0
        trending_used = [h for h in hashtags if h in trending]
        suggestions = []
        # If caption is very long
        if word_count > 40:
            suggestions.append("Consider shortening the caption to keep it concise.")
        # Long sentences may reduce readability
        if avg_sent_len > 20:
            suggestions.append("Break long sentences into shorter ones for better readability.")
        # Instagram allows up to 30 hashtags but recommends ~3–5【628487384572305†L442-L449】
        if len(hashtags) > 5:
            suggestions.append("Limit hashtags to 3–5 as recommended by Instagram guidelines.")
        # Encourage using trending/niche tags
        if not trending_used:
            suggestions.append("Consider using trending or niche‑specific hashtags to increase reach.")
        # Encourage adding at least a couple of hashtags
        if len(hashtags) < 2:
            suggestions.append("Add a few relevant hashtags to improve discoverability.")
        results.append({
            "Post": caption,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_len, 2),
            "avg_sentence_length": round(avg_sent_len, 2),
            "hashtag_count": len(hashtags),
            "trending_hashtags_used": ", ".join(trending_used) if trending_used else "None",
            "suggestions": "; ".join(suggestions) if suggestions else "None"
        })
    return pd.DataFrame(results)


def main():
    # Load data
    df = load_data("instagram_sample_data.csv")
    # Compute engagement rate
    df_eng = compute_engagement(df)
    # Identify top posts
    top_posts = df_eng.sort_values(by="EngagementRate", ascending=False)
    # Hashtag frequencies
    hashtag_df = analyze_hashtags(df_eng)
    # Best posting times
    hourly = analyze_posting_times(df_eng)
    # QC checks
    qc_df = qc_check(df_eng)

    print("\nOriginal dataset with engagement rates:\n")
    print(df_eng.to_string(index=False))

    print("\nTop posts by engagement rate (descending):\n")
    for _, row in top_posts.iterrows():
        print(f"- EngagementRate: {row['EngagementRate']:.4f} | Post: {row['Post'][:60]}...")

    print("\nHashtag usage frequency:\n")
    for _, row in hashtag_df.iterrows():
        print(f"# {row['Hashtag']}: {row['Count']} times")

    print("\nAverage engagement rate by posting hour:\n")
    for _, row in hourly.iterrows():
        print(f"Hour {int(row['Hour']):02d}: {row['EngagementRate']:.4f}")

    print("\nContent quality checks and suggestions:\n")
    for _, row in qc_df.iterrows():
        print(f"Post: {row['Post'][:60]}...")
        print(f"  Words: {row['word_count']}, Sentences: {row['sentence_count']}, "
              f"Avg word length: {row['avg_word_length']}, "
              f"Avg sentence length: {row['avg_sentence_length']}")
        print(f"  Hashtags used: {row['hashtag_count']}, Trending used: {row['trending_hashtags_used']}")
        print(f"  Suggestions: {row['suggestions']}")
        print("")


if __name__ == "__main__":
    main()
