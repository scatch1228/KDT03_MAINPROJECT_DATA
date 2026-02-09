from redis import Redis

try:
    r = Redis(host='10.125.121.184', port=6379)
    print("연결 시도 중...")
    if r.ping():
        print("연결 성공! (PONG)")
except Exception as e:
    print(f"연결 실패: {e}")

# python test_redis.py 실행해서 연결 성공! (PONG) 나오면 연결 성공.