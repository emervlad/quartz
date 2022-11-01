# QuartzNet5x5-CTC
* Поскольку по архитектуре separable всегда подразумевает r = 1, решил просто лишний раз не дублировать код.
* С SoundFile для OPUS была замечена проблема (supported file format but file is malformed), описанная с решением [здесь](https://github.com/bastibe/python-soundfile/issues/252).
* Проверить число параметров:
```bash
$ python nop.py
```
* Чтобы посчитать WER'ы на датасетах, необходимо запустить 
```bash
$ python main.py model.init_weights=<absolute_path>/data/q5x5_ru_stride_4_crowd_epoch_4_step_9794.ckpt`
```
с раскомментированными строчками trainer.validate(...). val_dataloader соответствует farfield'у, val_dataloader2 - crowd'у. Log'и для них лежат в 15-10-48 и 15-12-08 соотвественно
* Манифест для 100 семплов находится в файле mock_manifest.jsonl. Чтобы продемонстрировать на нём переобучение, достаточно заменить исходный manifest.jsonl на последний в YAML файле.
Графики с чекпоинтом доступны в подпапке outputs/2022-11-01/14-35-57/
![alt text](https://github.com/emervlad/quartz/blob/master/quartz_overfitting.png?raw=true)
