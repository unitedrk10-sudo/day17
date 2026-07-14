import os
from dotenv import load_dotenv
import pymupdf
# from openai import OpenAI  # [수정] LangChain 사용으로 OpenAI 직접 호출 제거
from langchain_openai import ChatOpenAI  # [수정] LangChain OpenAI 채팅 모델
from langchain_core.messages import SystemMessage  # [수정] LangChain 메시지 타입
from langchain_core.tools import tool  # [수정] 5.2 실습: 함수를 LLM tool로 등록

load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')  # [수정] ChatOpenAI가 .env의 OPENAI_API_KEY를 자동 사용
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # [수정] client 대신 llm 사용

# [수정] 요약에 사용할 LangChain LLM 인스턴스 (노트북 5.1·5.2와 동일 패턴)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def pdf_to_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ''
    for page in doc:
        text = page.get_text()
        full_text += text + '\n------------------------\n'

    pdf_file_name = os.path.basename(pdf_path)
    pdf_file_name = os.path.splitext(pdf_file_name)[0]
    txt_file_path = os.path.join(os.path.dirname(pdf_path), f'{pdf_file_name}.txt')
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return txt_file_path

def summarize_txt(txt_file_path):  # 내부용: @tool 아님 (summarize_pdf에서 직접 호출)
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    system_prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 

    작성해야 하는 포맷은 다음과 같음
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)

    ## 저자 소개


    ============= 이하 텍스트 ================
    {txt[:10000]}

    '''

    # [수정] client.chat.completions.create → llm.invoke
    response = llm.invoke([SystemMessage(content=system_prompt)])
    # [수정] response.choices[0].message.content → response.content
    return response.content

@tool  # [수정] 5.2 실습: summarize_pdf를 LLM이 호출할 수 있는 tool로 등록
def summarize_pdf(pdf_path: str) -> str:
    """PDF 파일을 텍스트로 변환한 뒤 요약문을 생성한다.

    Args:
        pdf_path (str): 요약할 PDF 파일 경로
    """
    txt_file_path = pdf_to_text(pdf_path)
    summary = summarize_txt(txt_file_path)
    summary_file_name = os.path.splitext(os.path.basename(pdf_path))[0] + '_summary.txt'
    summary_file_path = os.path.join(os.path.dirname(pdf_path), summary_file_name)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    return summary



if __name__ == "__main__":
    pdf_path = os.path.join(os.getcwd(),"samples/Language_models.pdf")
    summary = summarize_pdf.invoke({"pdf_path": pdf_path})  # [수정] tool 등록 후 invoke로 호출
    print(summary)
    print("성공적으로 요약되었습니다.")
