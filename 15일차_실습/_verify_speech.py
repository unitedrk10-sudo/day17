import time
from pydantic import BaseModel
from typing import List, Optional


class ClassState(BaseModel):
    students: List[str]
    boring_count: int
    last_speach: Optional[str] = None


def speech(state: ClassState) -> dict:
    sp = "안녕하세요. 실습 진행을 맡은 정한솔입니다."
    print(sp)
    return {
        "last_speach": sp,
        "boring_count": state.boring_count + 1,
    }


def rest(state: ClassState) -> dict:
    print("휴식 시간입니다.")
    time.sleep(0)
    return {
        "boring_count": 0,
    }


state = ClassState(students=["홍길동"], boring_count=5)
result = speech(state)
print("speech result:", result)
print("rest result:", rest(state))
