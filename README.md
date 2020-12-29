# Dataset of Generation From Knowledge

This repository provides the publicly available dataset for natural language generation from a knowledge.

이 데이터셋은 [KBox](http://kbox.kaist.ac.kr/)에서 추출한 RDF 트리플 형태의 지식을 대상으로 제작되었습니다. 

##제작 방법
이 데이터셋은 크게 작업과 검수라는 2개의 과정으로 구축되었습니다.

### 작업 과정
제공된 정보 : 지식 그래프  
작업자들이 주어진 지식 그래프에서 중요한 부분을 선택하고 이를 포함하는 요약문 작성.  

#### 작업자 요구사항
* 주어진 지식에 알맞은 요약문 작성
	* 주어진 지식 중 덜 중요하거나 불필요하다고 생각된 것은 제외하고 요약문 작성.
	* 주어진 지식을 모두 표현해도 괜찮음.
	* 주어진 지식 중 작업자가 아는 사실과 다른 내용이 있을수도  있으나, 무시하고 작업.
* 요약문에 포함된 지식을 체크박스에 표시.
* 표층형 수정 허용 수준 준수.

#### 표층형 수정 허용 수준
* 표층형을 바꾸지 않고 되도록 주어진 표현 그대로 사용.
	* 허용 수준 : 띄어쓰기 수정, 의미가 동일한 표현.
	* 허용 불가 수준 : 같은 의미로 특정되지 않는 표현.
* 예) <손흥민, 직업, 축구 선수>
	* 올바른 요약 : “손흥민은 축구 선수이다.”
	* 허용 범위의 요약 : “손흥민은 축구를 하는 선수이다.”
	* 잘못된 요약 : “손 선수는 축구를 한다.” (정확히 누구를 지칭하는지 특정되지 않음)

	
	
### 검수 과정
제공된 정보 : 지식 그래프와 작업 과정에서 수집된 요약문  
3명의 검수자가 지식 그래프와 요약문의 관계, 완성도, 제약조건 준수 여부를 평가.

#### 검수자 요구사항
* 3개의 질문 1~6점으로 답변.
	* 작성된 요약문이 제공된 지식을 잘 요약하였는가? (Informativeness)
		* 주어진 지식 중 중요하다고 생각되는 지식만 잘 표현하였다고 생각될수록 높은 점수.
	* 문법상 문제가 없는가? (Quality)
		* 문법적으로 문제가 있을수록 낮은 점수.
	* 요약문이 원어민 작문 수준으로 자연스러운가? (Naturalness)
* 2개의 질문 O/X로 답변.
	* 표층형 수정 허용 수준을 준수했는가?
	* 요약문에서 확인되는 내용과 주어진 지식의 체크박스 표시가 일치하는가.

## 데이터 형식
모든 데이터는 JSON 형태로 제공됩니다.
* id: 파일 id
* triple: 주어진 지식 목록
	* Rep: 요약문에 포함되는 true, 아니면 false
	* s: RDF 트리플에서 subject
	* p: RDF 트리플에서 property
	* o: RDF 트리플에서 object
* entities: 주어진 지식에 나오는 개체 목록
	* ne_type: 개체명 타입
	* type: 디비피디아 타입
	* uri
* Text: 요약문
* Eval: 검수 결과  
	* Checkbox: 요약문에서 확인되는 내용과 선택된 지식의 일치여부  
	* Informativeness: 작성된 요약문이 제공된 지식을 잘 요약한 정도  
	* Naturalness: 자연스러움 정도  
	* Quality: 문법상의 품질
	* Restriction: 표층형 수정 허용 수준 준수
###Example

	
	{
	    "Eval": [
	        {
	            "Checkbox": true,
	            "Informativeness": 6,
	            "Naturalness": 6,
	            "Quality": 6,
	            "Restriction": true
	        },
	        {
	            "Checkbox": true,
	            "Informativeness": 6,
	            "Naturalness": 6,
	            "Quality": 6,
	            "Restriction": true
	        },
	        {
	            "Checkbox": true,
	            "Informativeness": 6,
	            "Naturalness": 6,
	            "Quality": 6,
	            "Restriction": true
	        }
	    ],
	    "Text": "아시시의 프란치스코는 이탈리아에서 사망하였다. 프란치스코의 출생지는 아시시이다. 움브리아 주는 이탈리아에 있다. 프란치스코의 직업은 교황이다. 아시시는 로마의 한 지역이다.",
	    "entities": {
	        "교황": {
	            "ne_type": "CV_POSITION",
	            "type": [
	                "http://dbpedia.org/ontology/Person",
	                "http://dbpedia.org/ontology/ChristianBishop",
	                "http://dbpedia.org/ontology/Agent",
	                "http://dbpedia.org/ontology/Cleric"
	            ],
	            "uri": "http://kbox.kaist.ac.kr/resource/교황"
	        },
	        "교황_프란치스코": {
	            "ne_type": "PS_NAME",
	            "type": [
	                "http://dbpedia.org/ontology/Person",
	                "http://dbpedia.org/ontology/Pope",
	                "http://dbpedia.org/ontology/ChristianBishop",
	                "http://dbpedia.org/ontology/Cleric",
	                "http://dbpedia.org/ontology/Agent"
	            ],
	            "uri": "http://kbox.kaist.ac.kr/resource/교황_프란치스코"
	        },
	        "로마": {
	            "ne_type": "LCP_COUNTRY",
	            "type": [
	                "http://dbpedia.org/ontology/Location",
	                "http://dbpedia.org/ontology/Settlement",
	                "http://dbpedia.org/ontology/PopulatedPlace",
	                "http://dbpedia.org/ontology/Place"
	            ],
	            "uri": "http://kbox.kaist.ac.kr/resource/로마"
	        },
	        "아시시": {
	            "ne_type": "LCP_CITY",
	            "type": [
	                "http://dbpedia.org/ontology/Location",
	                "http://dbpedia.org/ontology/Settlement",
	                "http://dbpedia.org/ontology/PopulatedPlace",
	                "http://dbpedia.org/ontology/Place"
	            ],
	            "uri": "http://kbox.kaist.ac.kr/resource/아시시"
	        },
	        "아시시의_프란치스코": {
	            "ne_type": "PS_NAME",
	            "type": [
	                "http://dbpedia.org/ontology/Person",
	                "http://dbpedia.org/ontology/Saint",
	                "http://dbpedia.org/ontology/Cleric",
	                "http://dbpedia.org/ontology/Agent"
	            ],
	            "uri": "http://kbox.kaist.ac.kr/resource/아시시의_프란치스코"
	        },
	        "움브리아_주": {
	            "ne_type": "LCP_PROVINCE",
	            "type": [
	                "http://dbpedia.org/ontology/Region",
	                "http://dbpedia.org/ontology/AdministrativeRegion",
	                "http://dbpedia.org/ontology/Location",
	                "http://dbpedia.org/ontology/Settlement",
	                "http://dbpedia.org/ontology/PopulatedPlace",
	                "http://dbpedia.org/ontology/Place"
	            ],
	            "uri": "http://kbox.kaist.ac.kr/resource/움브리아_주"
	        },
	        "이탈리아": {
	            "ne_type": "LCP_COUNTRY",
	            "type": [
	                "http://dbpedia.org/ontology/Settlement",
	                "http://dbpedia.org/ontology/PopulatedPlace",
	                "http://dbpedia.org/ontology/Place",
	                "http://dbpedia.org/ontology/Country"
	            ],
	            "uri": "http://kbox.kaist.ac.kr/resource/이탈리아"
	        }
	    },
	    "id": 1,
	    "pid": "0-11",
	    "triple": [
	        {
	            "Rep": true,
	            "o": "이탈리아",
	            "p": "사망지(deathPlace)",
	            "s": "아시시의_프란치스코"
	        },
	        {
	            "Rep": true,
	            "o": "아시시",
	            "p": "출생지(birthPlace)",
	            "s": "아시시의_프란치스코"
	        },
	        {
	            "Rep": true,
	            "o": "이탈리아",
	            "p": "국가(country)",
	            "s": "움브리아_주"
	        },
	        {
	            "Rep": true,
	            "o": "교황",
	            "p": "직업(occupation)",
	            "s": "교황_프란치스코"
	        },
	        {
	            "Rep": true,
	            "o": "로마",
	            "p": "~의부분(isPartOf)",
	            "s": "아시시"
	        }
	    ]
	}


## 데이터 통계
### raw
1차 : 799개  
2차 : 4,689개

### purified
raw의 1차, 2차 데이터에서 아래의 기준에 맞는 데이터(3,640개)만 정제하여 8:1:1 비율로 분할.  
* Checkbox : 세개의 검수 결과 모두 true
* Restriction : 2개 이상의 검수 결과가 true
* Informativeness : 점수 합 12 이상
* Naturalness : 점수 합 10 이상
* Quality : 점수 합 12 이상

## Licenses
* `CC BY-NC-SA` [Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/2.0/)
* If you want to commercialize this resource, [please contact to us](http://semanticweb.kaist.ac.kr/)

## Publisher
[Machine Reading Lab](http://semanticweb.kaist.ac.kr/) @ KAIST

## Contact
Kuntae Kim. `kuntaek@kaist.ac.kr`

## Acknowledgement
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2013-0-00109, WiseKB: Big data based self-evolving knowledge base and reasoning platform)
