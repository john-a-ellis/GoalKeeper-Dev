import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, timezone
import json
import hashlib

class RedditConversionTracker:
    """
    A wrapper for sending conversion events to the Reddit Ads Conversions API.
    
    Attributes:
        base_url (str): Base URL for the Reddit Conversions API
        ad_account_id (str): Reddit Ads Account ID
        access_token (str): OAuth access token for API authentication
    """
    
    BASE_URL = "https://ads-api.reddit.com/api/v2.0/conversions/events"
    
    def __init__(self, ad_account_id: str, access_token: str):
        """
        Initialize the RedditConversionTracker.
        
        Args:
            ad_account_id (str): Your Reddit Ads Account ID (Pixel ID)
            access_token (str): OAuth access token with conversion tracking scope
        """
        self.ad_account_id = ad_account_id
        self.access_token = access_token
    
    def _hash_value(self, value: str) -> str:
        """
        Hash a value using SHA-256.
        
        Args:
            value (str): Value to be hashed
        
        Returns:
            str: SHA-256 hashed value
        """
        return hashlib.sha256(value.encode('utf-8')).hexdigest()
    
    def _validate_event_time(self, event_at: Optional[Union[str, datetime]] = None) -> str:
        """
        Validate and format event timestamp.
        
        Args:
            event_at (Optional[Union[str, datetime]]): Timestamp of the event
        
        Returns:
            str: Formatted ISO 8601 timestamp
        """
        if event_at is None:
            event_at = datetime.now(timezone.utc)
        
        # Convert to datetime if string
        if isinstance(event_at, str):
            try:
                event_at = datetime.fromisoformat(event_at.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid timestamp format. Use ISO 8601 format.")
        
        # Ensure timezone is UTC
        if event_at.tzinfo is None:
            event_at = event_at.replace(tzinfo=timezone.utc)
        
        # Check timestamp is within 7 days
        if (datetime.now(timezone.utc) - event_at).days > 7:
            raise ValueError("Event timestamp cannot be older than 7 days.")
        
        return event_at.isoformat()
    
    def send_conversion_event(
        self, 
        event_type: str, 
        event_at: Optional[Union[str, datetime]] = None,
        test_mode: bool = False,
        click_id: Optional[str] = None,
        custom_event_name: Optional[str] = None,
        event_metadata: Optional[Dict] = None,
        user_data: Optional[Dict] = None
    ) -> Dict:
        """
        Send a conversion event to the Reddit Ads API.
        
        Args:
            event_type (str): Type of conversion event (e.g., 'Purchase', 'AddToCart', 'Custom')
            event_at (Optional[Union[str, datetime]]): Timestamp of the event
            test_mode (bool, optional): Whether to run in test mode. Defaults to False.
            click_id (Optional[str], optional): Reddit click ID for attribution
            custom_event_name (Optional[str], optional): Name for custom events
            event_metadata (Optional[Dict], optional): Additional event details
            user_data (Optional[Dict], optional): User identification and matching data
        
        Returns:
            Dict: API response
        """
        # Prepare event payload
        event_payload = {
            "test_mode": test_mode,
            "events": [{
                "event_at": self._validate_event_time(event_at),
                "event_type": {
                    "tracking_type": event_type
                }
            }]
        }
        
        # Add click ID if provided
        if click_id:
            event_payload["events"][0]["click_id"] = click_id
        
        # Handle custom events
        if event_type == "Custom":
            if not custom_event_name:
                raise ValueError("Custom event requires a custom_event_name")
            event_payload["events"][0]["event_type"]["custom_event_name"] = custom_event_name
        
        # Add event metadata
        if event_metadata:
            event_payload["events"][0]["event_metadata"] = event_metadata
        
        # Add user data with optional hashing
        if user_data:
            processed_user_data = {}
            hash_fields = ['email', 'ip_address', 'external_id', 'aaid', 'idfa']
            
            for key, value in user_data.items():
                if key in hash_fields and value:
                    processed_user_data[key] = self._hash_value(str(value))
                else:
                    processed_user_data[key] = value
            
            event_payload["events"][0]["user"] = processed_user_data
        
        # Send API request
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.BASE_URL}/{self.ad_account_id}"
        print(f"This is the url: {url}")
        print(f"this is the headers: {headers}")
        try:
            response = requests.post(url, headers=headers, json=event_payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
    
    @classmethod
    def send_multiple_conversion_events(
        cls, 
        ad_account_id: str, 
        access_token: str, 
        events: List[Dict], 
        test_mode: bool = False
    ) -> Dict:
        """
        Send multiple conversion events in a single API call.
        
        Args:
            ad_account_id (str): Reddit Ads Account ID (Pixel ID)
            access_token (str): OAuth access token
            events (List[Dict]): List of conversion events
            test_mode (bool, optional): Whether to run in test mode. Defaults to False.
        
        Returns:
            Dict: API response
        """
        if len(events) > 1000:
            raise ValueError("Maximum 1,000 events can be sent per request")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        url = f"{cls.BASE_URL}/{ad_account_id}"
        
        payload = {
            "test_mode": test_mode,
            "events": events
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        
# Example usage
# def example_conversion_tracking():
#     """
#     Example demonstration of how to use the RedditConversionTracker
#     """
#     # Replace with your actual credentials
#     tracker = RedditConversionTracker(
#         ad_account_id=os.environ["REDDIT_PIXEL_ID"],
#         access_token=os.environ["GOALKEEPER_CONVERSION"]
#     )
    
  
#     # Custom event tracking
#     custom_event_response = tracker.send_conversion_event(
#         event_type="Custom",
#         test_mode = True,
#         custom_event_name="Testing Event",
#         user_data={
#             "email": "signup@example.com"
#         }
#     )
#     print(custom_event_response)

# if __name__ == "__main__":
#     example_conversion_tracking()

#    # Simple purchase conversion
#     purchase_response = tracker.send_conversion_event(
#         event_type="Purchase",
#         event_metadata={
#             "value_decimal": 49.99,
#             "currency": "USD",
#             "item_count": 1,
#             "conversion_id": "unique_purchase_id"
#         },
#         user_data={
#             "email": "customer@example.com",
#             "ip_address": "192.168.1.1",
#             "user_agent": "Mozilla/5.0 ...",
#             "screen_dimensions": {
#                 "width": 1920,
#                 "height": 1080
#             }
#         }
#     )
#     print(purchase_response)
  