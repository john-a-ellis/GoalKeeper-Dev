from app import app

# The server variable must be named 'application' for WSGI servers
application = app.server

# Some WSGI servers also look for 'server'
server = application

# # Optional: If you want to run the file directly
if __name__ == '__main__':
    app.run()