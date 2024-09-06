from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline, AutoTokenizer
from urllib.parse import urlparse, parse_qs
from dash import html, dash_table
import dash_bootstrap_components as dbc
from youtube_comment_downloader import *
import re
from datetime import datetime 
import pandas as pd

# sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", max_length = 512, truncation=True, padding=True)
# summarization_pipline = pipeline()
def ds_to_html_table(ds, title, title2):
    table_header=[
        html.Thead(html.Tr([html.Th(title), html.Th(title2)]))
    ]
    table_rows = []
    for i in range(len(ds)):
        if isinstance(ds.iloc[i], (int, float)):
            table_rows.append(html.Tr([
                html.Td([ds.index.to_list()[i].replace("_"," ").title()]),
                html.Td(str(ds.iloc[i]), style={'textAlign': 'right', 'display': 'table-cell'})
                ]))
        else:
            table_rows.append(html.Tr([
                html.Td(ds.index.to_list()[i].replace("_", " ").title()),
                html.Td(ds.iloc[i])
            ]))
    table_body = [html.Tbody(table_rows)]
    table = dbc.Table(table_header + table_body, bordered=True, id=title)
  
    return(table)
        
def df_to_html_table(df, mytitle, id):
    # Get the index label
    index_label = df.index.name
    
    # Create the table header
    table_header = [html.Thead(html.Tr([html.Th(mytitle, colSpan = str(len(df.columns) + 1))]))] + \
    [html.Thead(html.Tr([html.Th(index_label.title())] + [html.Th(col.title()) for col in df.columns]))]
    
    # Create the table rows
    table_rows = [
        html.Tr(
            [html.Td(df.index[i])] + 
            [
                html.Td(f"{df.iloc[i][col]:.2f}" if isinstance(df.iloc[i][col], float) and abs(df.iloc[i][col]) % 1 >= 0.01 else df.iloc[i][col])
                for col in df.columns
            ]
        )
        for i in range(len(df))
    ]
    table_body = [html.Tbody(table_rows)]
    # Create the table
    table = dbc.Table(table_header + table_body, bordered=True, id=id)
    
    return table


def convert_duration(duration):
    pattern = r'PT(?P<minutes>\d+M)?(?P<seconds>\d+S)?'
    match = re.match(pattern, duration)
    
    minutes = int(match.group('minutes')[:-1]) if match.group('minutes') else 0
    seconds = int(match.group('seconds')[:-1]) if match.group('seconds') else 0
    
    total_seconds = minutes * 60 + seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def analyze_comments(comments):
    #accepts a list of comments for which the sentiment is derived with a probability score (intensity score)
    results = sentiment_pipeline(comments)
    return results

def video_comments_to_dataframe(video_url):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    myResults = []
    final_df = None
    for comment in comments:
        myResults.append({
            "text": comment["text"],
            "replies": comment.get("replies", 0),
            "reply": comment.get("reply", False),
            "time": datetime.fromtimestamp(comment["time_parsed"]),
            "votes": comment.get("votes", 0),
            "cid":comment["cid"],
            "heart": comment.get("heart", 0),
        })
        
    for i in range(0, len(myResults), 5):
        chunk = [item["text"] for item in myResults[i:i+5]]
        scored_results = analyze_comments(chunk)    
        df = pd.DataFrame({
            'CID': [item["cid"] for item in myResults[i:i+5]],
            'Comment': chunk,
            'Sentiment': [r['label'] for r in scored_results],
            'Score': [r['score'] for r in scored_results],
            'Video': video_url,
            'Video_ID': get_video_id(video_url),
            'Replies': [item["replies"] for item in myResults[i:i+5]],
            'Reply': [item["reply"] for item in myResults[i:i+5]],
            'Heart': [item["heart"] for item in myResults[i:i+5]],
            'Time': [item["time"] for item in myResults[i:i+5]],
            'Votes': [item["votes"] for item in myResults[i:i+5]]
        })
        final_df = pd.concat([final_df, df], ignore_index=True)
    return final_df


def get_video_id (video_url):
    parsed_url = urlparse(video_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params['v'][0]
    return video_id

def get_channel_id_from_url(channel_url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/channel\/([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/user\/([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/c\/([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/@([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, channel_url)
        if match:
            return match.group(1)
    
    return None

def get_channel_id(youtube, channel_url):
    print(f"Attempting to get channel ID for URL: {channel_url}")
    channel_handle = get_channel_id_from_url(channel_url)
    
    try:
        if channel_handle and channel_handle.startswith('UC'):
            print(f"Extracted channel ID from URL: {channel_handle}")
            return channel_handle
        
        print("Searching for channel...")
        response = youtube.search().list(
            part="id,snippet",
            q=channel_url,
            type="channel",
            maxResults=1
        ).execute()
        
        if response['items']:
            channel_id = response['items'][0]['id']['channelId']
            print(f"Found channel ID via search: {channel_id}")
            return channel_id
        else:
            print("No channel found via search.")
            
        # If we still don't have a channel ID, try to get it from the channel handle
        if channel_handle:
            print(f"Attempting to get channel ID from handle: {channel_handle}")
            response = youtube.channels().list(
                part="id",
                forHandle=channel_handle.lstrip('@')
            ).execute()
            
            if response['items']:
                channel_id = response['items'][0]['id']
                print(f"Found channel ID from handle: {channel_id}")
                return channel_id
    
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred while searching for channel: {e.content}")
    
    print("Could not find the channel ID.")
    return None

def get_channel_statistics(channel_url, API_KEY):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    channel_id= get_channel_id(youtube, channel_url)
    if not channel_id:
        print("Could not find the channel ID. Exiting.")
        return {}
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=channel_id
    )
    response = request.execute()
    channel_stats = {
        'channel_name': response['items'][0]['snippet']['title'],
        'subscriber_count': response['items'][0]['statistics']['subscriberCount'],
        'view_count': response['items'][0]['statistics']['viewCount'],
        'video_count': response['items'][0]['statistics']['videoCount']
    }

    return channel_stats
    
    

def get_channel_videos(channel_url, API_KEY):
    print(f"Initializing YouTube API client...")
    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    channel_id = get_channel_id(youtube, channel_url)
    if not channel_id:
        print("Could not find the channel ID. Exiting.")
        return {}
    
    videos = {}
    next_page_token = None
    
    print(f"Starting to fetch videos for channel ID: {channel_id}")
    while True:
        try:
            print(f"Fetching page of results. Token: {next_page_token}")
            res = youtube.search().list(
                channelId=channel_id,
                type="video",
                part="id,snippet",
                order="date",
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in res['items']]
            
            # Fetch additional details for these videos
            video_response = youtube.videos().list(
                part="snippet,statistics,contentDetails,status",
                id=','.join(video_ids)
                ).execute()
    
            for item in video_response['items']:
                video_id = item['id']
                snippet = item['snippet']
                statistics = item['statistics']
                content_details = item['contentDetails']
                status = item['status']
        
                videos[video_id] = {
                    'url': 'https://www.youtube.com/watch?v=' + video_id,
                    'title': snippet['title'],
                    # 'description': snippet['description'],
                    'publish_date': datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                    'view_count': int(statistics.get('viewCount', 0)),
                    'like_count': int(statistics.get('likeCount', 0)),
                    'dislike_count': int(statistics.get('dislikeCount', 0)),
                    'favourite_count': int(statistics.get('favouriteCount', 0)),
                    'comment_count': int(statistics.get('commentCount', 0)),
                    'duration': convert_duration(content_details['duration']),
                    'watch_time': int(statistics.get('watchTimeMinutes',0)),
                    # 'definition': content_details['definition'],
                    'privacy_status': status['privacyStatus'],
                    # 'license': status['license'],
                    'tags': snippet.get('tags', []),
                    # 'category_id': snippet['categoryId']
                }

            
            print(f"Fetched {len(video_ids)} videos. Total so far: {len(videos)}")
            
            next_page_token = res.get('nextPageToken')
            
            if not next_page_token:
                print("No more pages to fetch.")
                break
        
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred while fetching videos: {e.content}")
            break
    
    return videos

def create_columnDefs(df):
    columnDefs = []
    for col in df.columns:
        column_def = {
            "field": col,
            "headerName": col.replace('_', ' ').title(),
            "filter": True,
            "sortable": True
        }
        
        # Determine the column type and set appropriate properties
        if df[col].dtype == 'object':
            column_def["filter"] = "agTextColumnFilter"
        elif df[col].dtype in ['int64', 'float64']:
            column_def["filter"] = "agNumberColumnFilter"
            column_def["type"] = "numericColumn"
        elif df[col].dtype == 'bool':
            column_def["filter"] = "agSetColumnFilter"
            column_def["cellRenderer"] = "agBooleanCellRenderer"
        elif df[col].dtype in ['datetime64[ns]','timedelta64[ns]']:
            column_def["filter"] = "agDateColumnFilter"
            column_def["type"] = "dateColumn"
        
        columnDefs.append(column_def)
    
    return columnDefs

def convert_duration(duration):
    pattern = r'PT(?P<minutes>\d+M)?(?P<seconds>\d+S)?'
    match = re.match(pattern, duration)
    
    minutes = int(match.group('minutes')[:-1]) if match.group('minutes') else 0
    seconds = int(match.group('seconds')[:-1]) if match.group('seconds') else 0
    
    total_seconds = minutes * 60 + seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def fetch_subscriber_data(api_key, channel_id):
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.subscriptions().list(
        part='snippet',
        channelId=channel_id,
        maxResults=100
    )
    response = request.execute()

    subscriber_count = 0
    subscriber_changes = []

    for item in response['items']:
        subscriber_count += 1
        subscriber_changes.append({
            'subscriber_id': item['snippet']['resourceId']['channelId'],
            'subscriber_name': item['snippet']['title'],
            'subscription_date': item['snippet']['publishedAt']
        })

    return subscriber_count, subscriber_changes
