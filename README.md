## 음성 인식 차량 정보 시스템

이 프로젝트는 음성 인식을 통해 차량 정보를 제공하고 음악을 재생할 수 있는 시스템입니다. 사용자는 음성 명령을 통해 특정 차량에 대한 정보를 조회하거나 음악을 재생할 수 있습니다. 이 시스템은 PDF 문서에서 데이터를 추출하여 데이터베이스에 저장하고, 이를 기반으로 질문에 답변합니다.

### 주요 기능

1. **음성 인식**: 음성을 통해 명령을 입력받습니다.
2. **차량 정보 제공**: 데이터베이스에 저장된 차량 정보에 대해 답변합니다.
3. **음악 재생**: 유튜브에서 음악을 검색하여 재생합니다.
4. **텍스트 음성 변환**: 텍스트를 음성으로 변환하여 응답합니다.
5. **로그 기록**: 사용자 질문과 응답을 로그 파일에 기록합니다.

### 설치 및 실행 방법

#### 사전 요구 사항

- Python 3.9.13 이상
- 필요한 라이브러리 설치: `pip install -r requirements.txt`
- OpenAI API 키 및 Google API 키 필요

#### 설치

1. 이 저장소를 클론합니다.

```bash
git clone https://github.com/your-repo/voice-vehicle-info-system.git
cd voice-vehicle-info-system
```

2. 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

3. 환경 변수 설정

- `OPENAI_API_KEY`와 `GOOGLE_API_KEY` 환경 변수를 설정합니다.

4. 폴더 구조 생성

- 로그 폴더 생성: `../log/`
- PDF 파일 폴더 생성: `../pdfs/`
- Chroma 데이터베이스 폴더 생성: `../car_info_sql/`
- 출력 파일 폴더 생성: `../output/`
- 사운드 파일 폴더 생성: `../sounds/`

#### 실행

Jupyter Notebook을 실행하여 각 셀을 순차적으로 실행합니다.

### 사용 방법

#### 주요 셀 설명

1. **환경 변수 설정 및 로그 설정**
   - `os.environ["OPENAI_API_KEY"]`와 `os.environ["GOOGLE_API_KEY"]`를 통해 API 키 설정
   - 로그 파일 설정 및 로그 함수 `log_query_and_response` 구현

2. **PDF 텍스트 및 테이블 처리**
   - `extract_tables_from_pdfplumber` 함수: PDF 파일에서 테이블 데이터를 추출
   - `process_pdfs` 함수: PDF 파일들을 처리하여 문서 리스트로 반환

3. **ChromaDB 벡터스토어 생성**
   - PDF 데이터를 벡터스토어로 저장하고, 이미 존재하는 경우 이를 호출

4. **OpenAI API 체이닝**
   - `ChatPromptTemplate`을 통해 사용자 입력을 처리하고, LLMChain을 사용하여 응답 생성

5. **텍스트 처리 및 음성 변환**
   - `convert_text` 함수: 텍스트 내 특정 단어를 변환
   - `text_to_speech` 함수: 텍스트를 음성으로 변환하여 재생

6. **음악 검색 및 재생**
   - `search_music` 함수: 유튜브에서 음악을 검색
   - `download_audio` 함수: 유튜브에서 오디오를 다운로드
   - `play_music_thread` 함수: 오디오 재생을 위한 스레드 실행
   - `play_music_request` 함수: 음악 재생 요청 처리

7. **음성 인식 및 처리**
   - `listen_and_process` 함수: 음성 명령을 인식하고 처리
   - `stop_listening` 함수: 음성 인식 중지
   - `stop_music` 함수: 음악 재생 중지

8. **GUI 설정**
   - Tkinter를 사용하여 GUI를 설정하고, 각 버튼에 기능 연결

#### 명령 예시

- 차량 정보 조회: "기아 K3의 제원 알려줘"
- 음악 재생: "비틀즈의 노래 틀어줘"

### 기타

- 추가적으로 개선할 사항이나 오류가 있을 경우 이슈를 등록해주세요.
- 기여를 원하시는 분들은 풀 리퀘스트를 제출해주세요.

이 프로젝트에 대해 질문이나 문제가 있을 경우, 리포지토리의 이슈 트래커를 이용해주세요.

### Note

- Jupyter Notebook을 실행할 때, 각 셀을 순차적으로 실행하여 코드가 정상적으로 동작하는지 확인하세요.
- 음성 인식 기능을 사용할 때, 적절한 마이크가 연결되어 있어야 합니다.
- 유튜브 음악 재생 기능을 사용할 때, 네트워크 연결이 필요합니다.
