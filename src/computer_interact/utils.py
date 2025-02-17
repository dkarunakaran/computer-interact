import logging
import yaml
import os.path
#from google.auth.transport.requests import Request
#from google.oauth2.credentials import Credentials
#from google_auth_oauthlib.flow import InstalledAppFlow
from PIL import Image, ImageDraw, ImageColor

# Ref - https://medium.com/pythoneers/beyond-print-statements-elevating-debugging-with-python-logging-715b2ae36cd5
def logger_helper(cfg):

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Capture all messages of debug or higher severity

    ### File handler for errors
    # Create a file handler that writes log messages to 'error.log'
    file_handler = logging.FileHandler('error.log') 
    # Set the logging level for this handler to ERROR, which means it will only handle messages of ERROR level or higher
    file_handler.setLevel(logging.ERROR)  

    ### Console handler for info and above
    # Create a console handler that writes log messages to the console
    console_handler = logging.StreamHandler()  
    
    if cfg.debug == True:
        console_handler.setLevel(logging.DEBUG)  
    else:
        # Set the logging level for this handler to INFO, which means it will handle messages of INFO level or higher
        console_handler.setLevel(logging.INFO)  

    ### Set formats for handlers
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
    # Apply the formatter to the file handler 
    file_handler.setFormatter(formatter) 
    # Apply the formatter to the console handler
    console_handler.setFormatter(formatter)  

    ### Add handlers to logger
    # Add the file handler to the logger, so it will write ERROR level messages to 'error.log'
    logger.addHandler(file_handler)  
    # Add the console handler to the logger, so it will write INFO level messages to the console
    logger.addHandler(console_handler)  

    # Now when you log messages, they are directed based on their severity:
    #logger.debug("This will print to console")
    #logger.info("This will also print to console")
    #logger.error("This will print to console and also save to error.log")

    return logger

"""        
def google_api_authenticate(cfg):
    
    if not os.environ.get("GOOGLE_API_CREDS_LOC"):
      raise KeyError("Local file path of credentials.json is missing, please provide it in .env file.") 

    # If modifying these scopes, delete the file token.json.
    SCOPES = cfg['GOOGLE_API']['scopes']

    # Authenticate
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
      creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
      else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0, open_browser=True)
      # Save the credentials for the next run
      with open("token.json", "w") as token:
        token.write(creds.to_json())

    return creds
"""

def draw_point(image: Image.Image, point: list, color=None):
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  
        except ValueError:
            color = (255, 0, 0, 128)  
    else:
        color = (255, 0, 0, 128)  

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color
    )
    
    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [(x - center_radius, y - center_radius), 
         (x + center_radius, y + center_radius)],
        fill=(0, 255, 0, 255)
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')

