from flask import Flask

app = Flask(__name__)

@app.route(/)
def hello():
    return "Hello, World! This is a test Flask app."

@app.route('/health')
def health():
    return {status": "healthy, message": Minimal Flask app is running}

if __name__ == "__main__":
    print("Starting minimal Flask test app...")
    print(This should start once and stay running without restarts.")
    app.run(
        debug=False,  # Disable debug mode
        use_reloader=False,  # Disable reloader
        host=0.0.00,
        port=5001  # Use different port to avoid conflicts
    ) 