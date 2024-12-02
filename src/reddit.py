import requests
import json
import uuid
from typing import Dict, Any, Optional
import logging
import hashlib

class RedditConversionsAPIClient:
    """
    Client for tracking user registrations via Reddit Conversions API v2.0
    """
    
    BASE_URL = "https://ads-api.reddit.com/api/v2.0/conversions/events"
    
    def __init__(self, 
                 account_id: str, 
                 conversion_access_token: str, 
                 pixel_id: str,
                 test_mode: bool = False):
        """
        Initialize the Reddit Conversions API client for user registration tracking
        
        :param account_id: Your Reddit Ads account ID
        :param conversion_access_token: Conversion access token
        :param pixel_id: Your Reddit pixel ID
        :param test_mode: Enable test mode for event tracking
        """
        self.account_id = account_id
        self.conversion_access_token = conversion_access_token
        self.pixel_id = pixel_id
        self.test_mode = test_mode
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _generate_event_id(self) -> str:
        """
        Generate a unique event ID
        
        :return: Unique event identifier
        """
        return str(uuid.uuid4())
    
    def _hash_identifier(self, identifier: str) -> str:
        """
        Hash user identifiers for privacy and API requirements
        
        :param identifier: Raw identifier to hash
        :return: SHA-256 hashed identifier
        """
        return hashlib.sha256(identifier.lower().encode('utf-8')).hexdigest()
    
    def send_user_registration_event(self, 
                                     user_id: str, 
                                     email: Optional[str] = None,
                                     additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a user registration conversion event
        
        :param user_id: Unique identifier for the user
        :param email: User's email address (optional)
        :param additional_data: Additional registration metadata
        :return: API response
        """
        # Prepare user identifiers
        user_data = {
            "extern_id": self._hash_identifier(user_id)
        }
        
        # Add hashed email if provided
        if email:
            user_data["em"] = self._hash_identifier(email)
        
        # Prepare event data
        event_data = {
            "user_data": user_data,
            "registration_method": "app"
        }
        
        # Add any additional metadata
        if additional_data:
            event_data.update(additional_data)
        
        # Send the conversion event
        return self.send_conversion_event("signup", event_data)
    
    def send_conversion_event(self, 
                               event_type: str, 
                               event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a conversion event to Reddit Ads API
        
        :param event_type: Type of conversion event
        :param event_data: Event details and user identifiers
        :return: API response
        """
        # Construct the full endpoint URL
        endpoint = f"{self.BASE_URL}/{self.account_id}"
        
        # Prepare the request payload
        payload = {
            "events": [{
                "event_type": event_type,
                "event_id": self._generate_event_id(),
                "pixel_id": self.pixel_id,
                **event_data
            }],
            "test_mode": self.test_mode
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.conversion_access_token}"
        }
        
        try:
            # Send the conversion event
            response = requests.post(
                endpoint, 
                headers=headers, 
                data=json.dumps(payload)
            )
            
            # Raise an exception for bad responses
            response.raise_for_status()
            
            # Log successful event
            log_mode = "TEST" if self.test_mode else "PRODUCTION"
            self.logger.info(f"User registration event sent successfully ({log_mode}): {event_type}")
            
            return response.json()
        
        except requests.RequestException as e:
            # Log and re-raise the exception
            self.logger.error(f"Failed to send user registration event: {e}")
            raise RuntimeError(f"User registration event sending failed: {e}")

# Example usage in an app registration flow
if __name__ == "__main__":
    # Replace with your actual credentials
    ACCOUNT_ID = "your_account_id"
    CONVERSION_ACCESS_TOKEN = "your_conversion_access_token"
    PIXEL_ID = "your_pixel_id"
    
    # Initialize the client
    reddit_client = RedditConversionsAPIClient(
        account_id=ACCOUNT_ID,
        conversion_access_token=CONVERSION_ACCESS_TOKEN,
        pixel_id=PIXEL_ID,
        test_mode=True  # Set to False for production
    )
    
    def register_new_user(username: str, email: str) -> None:
        """
        Example user registration method
        
        :param username: User's unique username
        :param email: User's email address
        """
        try:
            # Simulate user registration process
            print(f"Registering user: {username}")
            
            # Send registration event to Reddit
            response = reddit_client.send_user_registration_event(
                user_id=username,
                email=email,
                additional_data={
                    "app_name": "TheGoalkeeper",
                    "registration_source": "mobile_app",
                    "user_country": "US"  # Optional additional context
                }
            )
            
            print("Registration event tracking successful:", response)
        
        except Exception as e:
            print(f"Error tracking user registration: {e}")
    
    # Example usage
    register_new_user("johndoe123", "john.doe@example.com")