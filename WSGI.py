import sys
import os

# Add the application directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/var/www/webroot/ROOT')
os.chdir('/var/www/webroot/ROOT')

from app import server as application

# # Optional: If you want to run the file directly
if __name__ == '__main__':
    app.run()