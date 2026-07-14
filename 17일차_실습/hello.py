from fastapi import FastAPI

app = FastAPI(title='Day17 Demo', version='0.1.0')

@app.get('/hello')
def hello():
    return {'message': '안녕하세요. 서버 설정에 성공하셨네요. 축하합니다.'}

@app.get('/')
def welcome():
    return {'message': 'hello world'}