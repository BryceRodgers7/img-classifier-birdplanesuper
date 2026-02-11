"""
Search Queries for Image Classification Dataset

Defines search queries for each category in the classifier.
"""

SEARCH_QUERIES = {
    'bird': [
        'bird flying',
        'crow',
        'eagle',
        'parrot',
        'pidgeon',
    ],
    'plane': [
        'airplane flying',
        'airplane',
        'jet plane',
        'prop plane',
        'plane landing'
    ],
    'superman': [
        'superman flying',
        'superman',
        'superman costume',
        'superman comic',
        'superman live-action',
    ],
    'other': [
        'person',
        'crowd',
        'portrait',
        'face',
        'city',
        'street',
        'building',
        'skyscraper',
        'bridge',
        'interior',
        'room',
        'kitchen',
        'office',
        'furniture',
        'spaceship',
        'car',
        'truck',
        'train',
        'helicopter',
        'boat',
        'motorcycle',
        'bicycle',
        'construction',
        'pattern',
        'bulldozer',
        'tractor',
        'landscape',
        'mountain',
        'forest',
        'desert',
        'beach',
        'ocean',
        'waterfall',
        'flowers',
        'trees',
        'snow',
        'rocks',
        'underwater',
        'fish',
        'insects',
        'reptile',
        'dog',
        'cat',
        'food',
        'restaurant',
        'painting',
        'illustration',
        'anime',
        'cartoon',
        'screenshot',
        'texture',
    ]
}


def load_search_queries():
    """Return the search queries dictionary."""
    return SEARCH_QUERIES