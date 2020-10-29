# Callbacks

## EarlyStopping

* 설정한 metric이 더 이상 증가하지 않을 때 훈련을 중단하는 함수
  * 적절한 Epoch를 조정하는데 도움을 줌
* Aliases

```python
from tf.keras.callbacks import EarlyStopping

ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                   baseline=None, restore_best_weights=False)
```

* 주요 Arguments
  * `moniter` : 성능 평가할 지표	ex) `val_loss`
  * `min_delta` : `min_delta` 미만의 변동은 성능 개선이 없는 것으로 판단
  * `patience` : 성능이 나아지지 않는 Epoch를 몇번이나 허용할 것인지 정의
  * `mode` : {"auto", "min", "max"} 중 하나. 지표에 따라 선택 사용
  * `baseline` : 설정한 값에 도달하면 훈련 중단



## ModelCheckpoint

* 성능 평가 중 가장 좋은 모델을 저장해주는 함수
  * 이전 Epoch에 비해 성능이 좋은 경우, 무조건 이 때의 parameter들을 저장
* Aliases

```python
from tf.keras.callbacks import ModelCheckpoint

MC = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
                     save_weights_only=False, mode='auto', save_freq='epoch', 
                     options=None, **kwargs)
```

* 주요 Arguments
  * `moniter` : 성능 평가할 지표	ex) `val_loss`
  * `save_best_only` : True 값이면 가장 성능이 좋은 모델을 남김
  * `mode` : {"auto", "min", "max"} 중 하나. 지표에 따라 선택 사용

