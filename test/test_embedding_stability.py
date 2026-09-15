import requests
import time

for i in range(5):
    start = time.time()

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": ["Where is my Amazon package?"] * 100
        },
        timeout=120
    )

    print(
        f"Test {i+1}: "
        f"Status={r.status_code}, "
        f"Time={time.time()-start:.2f}s"
    )

    if r.status_code != 200:
        print(r.text[:300])