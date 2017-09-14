"""
The app instance, imports and configures flask
"""
from flask import Flask
from .routes import ROUTES


app = Flask(__name__)
# this would be an ideal place to put prod configuration settings
app.config.from_envvar('PROD_SETTINGS', silent=True)

app.register_blueprint(ROUTES)
