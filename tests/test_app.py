from app import app

def test_health():
    c = app.test_client(); r = c.get('/health')
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

def test_home():
    c = app.test_client(); r = c.get('/')
    assert r.status_code == 200
    assert b"BioForm" in r.data
