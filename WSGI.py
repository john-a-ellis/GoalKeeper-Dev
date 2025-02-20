import sys
import os

# Add the application directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from app import app

# The server variable must be named 'application' for WSGI servers
application = app.server

# Some WSGI servers also look for 'server'
server = application

# # Optional: If you want to run the file directly
if __name__ == '__main__':
    app.run()