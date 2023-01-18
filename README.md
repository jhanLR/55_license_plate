# 차량 번호판 인식 모델

## 실행 방법

#### 차량 번호 인식 모델
Model	: Clova(TPS-ResNet-BiLSTM-Attn)<br>
Task	: License plate Recognition 
* train.py 실행
```
cd licenseplate_ocr
python train.py 	--train_data './input/lmdb_train'
			--valid_data './input/lmdb_valid'
		        --batch_size 192
    			--manualseed 2000
    			--Transformation 'TPS'
    			--FeatureExtraction 'ResNet'
    			--SequenceModeling 'BiLSTM'
    			--Prediction 'Attn'
```

###### train.py 파라미터 세부정보
```
--Transformation {Transformation 모듈 선택}
--FeatureExtraction {특징 추출 모듈 선택}
--SequenceModeling {시퀀스 모델링 모듈 선택}
--Prediction {예측 모듈 선택}
--image_folder {입력 이미지 폴더 경로}
--saved_model {학습 완료된 모델 정보 파일 경로}
```

* test.py 실행

```
python test.py 	--eval_data ./input/lmdb_test
    		--batch_size 192
    		--saved_model 
    		            './saved_models/TPS-ResNet-BiLSTM-Attn-Seed2000-Renewed/best_accuracy.pth'
    		--Transformation 'TPS'
    		--FeatureExtraction 'ResNet'
    		--SequenceModeling 'BiLSTM'
    		--Prediction 'Attn'
```
###### Test.py 파라미터 세부정보
```
--Transformation {Transformation 모듈 선택}
--FeatureExtraction {특징 추출 모듈 선택}
--SequenceModeling {시퀀스 모델링 모듈 선택}
--Prediction {예측 모듈 선택}
--image_folder {입력 이미지 폴더 경로}
--saved_model {학습 완료된 모델 정보 파일 경로}
```

#### 차량 번호판 탐지 모델
Model	: YOLO v4<br>
Task	: License plate Detection
* train.py 실행
```
cd licenseplate_det
python train.py --batch_size 16
        	--epochs 500
        	--project '/runs/train'
        	--name 'license_plate'
```

###### train.py 파라미터 세부정보
```
--batch_size {GPU에 따라 설정}
--epochs {학습 횟수}
--project {학습 로그 및 정보 저장 위치}
--name {프로젝트명 폴더 경로}
```

* test.py 실행

```
python test.py 	--batch_size 16
    		--weights '/runs/train/license_plate/best_overall.pt'
    		--project '/runs/test'
    		--name 'license_plate'
```



## Dependency
* 아래는 특정 버전을 위한 패키지를 표시
* 나머지 라이브러리 및 패키지는 requirements.txt에 명시
```
pip install -r requirements.txt
```
```
Python >= 3.9
opencv-python==4.6.0.66 
torch==1.11.0+cu113
torchaudio==0.11.0+cu113
torchinfo==1.7.0
torchsummary==1.5.1
torchvision==0.12.0+cu113
natsort >= 7.1.0
nltk >= 3.5
pillow >= 7.0.0
lmdb >= 1.0.0
```


## Reference
https://github.com/clovaai/deep-text-recognition-benchmark<br>
https://github.com/WongKinYiu/PyTorch_YOLOv4

## License
MIT License

Copyright (c) 2023 jhanLR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
