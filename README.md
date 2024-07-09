## 음성 인식 차량 정보 시스템

이 프로젝트는 음성 인식을 통해 차량 정보를 제공하고 음악을 재생할 수 있는 시스템입니다. 사용자는 음성 명령을 통해 특정 차량에 대한 정보를 조회하거나 음악을 재생할 수 있습니다. 이 시스템은 PDF 문서에서 데이터를 추출하여 데이터베이스에 저장하고, 이를 기반으로 질문에 답변합니다.

### 주요 기능

1. **음성 처리**: 음성을 통해 STT를 사용하여 명령을 입력받고 응답을 TTS를 통해 방송합니다.
2. **차량 정보 제공**: Langchain을 사용하여 차량 Owner's manual pdf 파일의 정보를 RAG를 사용하여 답변합니다.
3. **음악 재생**: 유튜브에서 음악을 검색하여 재생합니다.
4. **맛집 추천** : 서울 지역의 맛집을 추천해줍니다. (이번 버젼에는 '성수동'만 제공합니다.)

### 설치 및 실행 방법
- 추후 구글 플레이에서 'drivetalk'으로 검색하여 설치합니다.

### Repository 구성
- Flutter 앱 코드 : https://github.com/pladata-encore/DA34-final-trois-blackmamba_client_prod
- 앱 기본기능 API 코드 : https://github.com/pladata-encore/DA34-final-trois-blackmamba_api
- Langchain API 코드 : https://github.com/pladata-encore/DA34-final-trois-blackmamba_langchain
- Local Web 서버 코드 : https://github.com/pladata-encore/DA34-final-trois-blackmamba_web
