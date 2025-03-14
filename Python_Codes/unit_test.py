import pytest
from app import app 

@pytest.fixture
def client():
    """Create a test client without running Flask"""
    app.config['TESTING'] = True  
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test if the homepage loads successfully"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Chatbot" in response.data  

def test_ask_api(client):
    """Test chatbot API with a sample query"""
    response = client.post('/ask', json={"query": "What is Finance?"})
    assert response.status_code == 200  
    assert b"answer" in response.data  

def test_stock_api(client):
    """Test stock price API"""
    response = client.get('/stock/AAPL')
    assert response.status_code in [200, 400]  
