import os
from dotenv import load_dotenv
import pymupdf
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

def summarize_txt(txt_file_path):
    client = OpenAI(api_key=api_key)
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

    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        temperature = 0.1,
        messages=[
            {"role":"system","content":system_prompt},
        ]
    )

    return response.choices[0].message.content

def summarize_pdf(pdf_path):
    txt_file_path = pdf_to_text(pdf_path)
    summary = summarize_txt(txt_file_path)
    summary_file_name = os.path.splitext(os.path.basename(pdf_path))[0] + '_summary.txt'
    summary_file_path = os.path.join(os.path.dirname(pdf_path), summary_file_name)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    return summary



if __name__ == "__main__":
    pdf_path = os.path.join(os.getcwd(),"samples/Language_models.pdf")
    summary = summarize_pdf(pdf_path)
    print(summary)
    print("성공적으로 요약되었습니다.")


