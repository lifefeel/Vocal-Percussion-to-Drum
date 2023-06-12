# Vocal Percussion to Drum

딥러닝 기반 음악생성 과목 프로젝트 시연 코드입니다.

## 팀원

- 류지우
- 김대웅
- 이정필

## 환경

git 받아오기

```bash
git clone git@github.com:lifefeel/Vocal-Percussion-to-Drum.git
```

Python 가상환경 설치

```bash
cd Vocal-Percussion-to-Drum
virtualenv -p python3 myenv
source myenv/bin/activate
```

Python 패키지 설치

```
pip install -r requirements.txt
```

## 실행

### 개발 환경에서 실행하는 방법

웹 인터페이스 데모를 띄위기 위해 아래 코드를 실행합니다.

```bash
python main.py
```

실행을 하면 데모를 확인할 수 있는 웹 주소가 아래와 같이 뜹니다. 

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

### 외부에서 접속이 되도록 하는 방법

VSCode에서 실행 시 자동으로 포트포워딩이 되기 때문에 로컬 웹 브라우저에서 바로 접근이 가능하며, 서버환경에서 직접 실행 시에는 외부에서 접속이 가능하도록 설정하여야 합니다. 아래와 같이 share 옵션을 통해 외부에서 접속이 가능합니다.

```
python main.py --share true
```

 실행이 되면 아래와 같이 72시간 동안 접속이 가능한 URL이 활성화 됩니다. 

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://38eca9efbeb6c9b501.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
```

