
"""
Flask blueprint for Gateway API routes
contains answer_string
"""

from flask import Blueprint, jsonify
from .keras_worker import mnist_by_index

ROUTES = Blueprint('routes', __name__)


@ROUTES.route('/index/<index>', methods=['GET'])
def random(index):
    result = mnist_by_index(int(index))
    return jsonify(estimation=str(result[0]), represenatation=result[1].tolist())