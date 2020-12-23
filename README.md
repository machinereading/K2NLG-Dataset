# K2NLG
이 코드는 RDF 트리플 형태의 지식들로부터 요약문을 생성한다.  
전체 모듈은 콘텐트 선택기, [GraphWriter](https://github.com/rikdz/GraphWriter)를 기반으로 수정한 NLG, 조사 선택기로 이루어졌다.

## 데이터


## 콘텐트 선택기
콘텐트 선택기에서는 입력으로 받은 지식 집합 중 요약문에 표현할 지식으로만 구성된 부분 집합을 구하는 역할을 한다.  

학습:
  
	python content.py -content_mode train -content_model_save ../save/content.pt -datadir DATA_DIR -nlg_train_path NLG_TRAIN_PATH -nlg_dev_path NLG_DEV_PATH -nlg_test_path NLG_TEST_PATH
평가:

	python content.py -content_mode eval -content_model_save ../save/content.pt -datadir DATA_DIR -nlg_train_path NLG_TRAIN_PATH -nlg_dev_path NLG_DEV_PATH -nlg_test_path NLG_TEST_PATH

## NLG
NLG에서는 입력으로 받은 지식 집합을 한국어로 된 문장으로 변환하는 역할을 한다.


## 조사 선택기 