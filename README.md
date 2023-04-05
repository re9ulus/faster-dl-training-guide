#  Как учить Deep Learning модели быстрее

Рекомендации по ускорению тренировок в рамках одной GPU.

### Достань более быструю GPU
Как бы ты не ухищрялся, достичь производительности следующей GPU в модельном ряду, скорее всего, не получится.

Если есть возможность достать GPU получше - это лучшее решение.

### Бенчмаркай
Перед тем как начать что-то улучшать, научись это измерять.
Используй [профайлеры](https://pytorch.org/tutorials/recipes/recipes/benchmark.html).

Проверяй что тренировка не разваливается, после внесения изменений.

### Используй оффициальный докер образ NVidia
Использование Pytorch [docker-образа](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) от NVidia поможет избежать нетривиальных проблем с версией `CUDA`, неработающих тензорных ядер и множества других интересных проблем. C которыми ты почти наверняка столкнешься, пытаясь ускорить тренировку моделей.

Если есть возможность - лучше использовать.

### Проверь боттлнеки на чтение данных

#### DataLoader
По умолчанию `DataLoader` будет готовить данные для обучения в основном потоке исполнения.
Попробуй выставить `num_workers=1`, чтобы отселить загрузку данны в отдельные процесс.
С помощью профайлера и бенчмарков определи оптимальное число воркеров. 

#### pin_memory + non_blocking
Параметр `DataLoader(pin_memory=True)` в связке с `Tensor.to(device, non_blocking=True)`, может ускорить загрузку тензоров на GPU.

#### Быстрые форматы ридеры
Если упираешься в чтение данных - поищи более быстрые форматы / ридеры под свои данные.
- [Рекомендации для картинок](https://fastai1.fast.ai/performance.html#faster-image-processing).

### Убирай синхронизацию
Любые вызовы `Tensor.cpu()/to()/numpy()` заставляют GPU синхронизироваться.

Синхронизация это дорого.
Выкидывай все ненужные синхронизации.

### Батч сайз

#### Увеличивай батч-сайз
Увеличение размера батча делает твою тренировку стабильнее (если ты не занимаешься совсем эзотерикой) и позволяет протолкнуть чуть больше данных в единицу времени. [Подробнее](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size).

Не забывай про [эвристики](https://stackoverflow.com/a/53046624) связывающие размет батча и learnin-rate.

#### Фиксируй размер батч-сайза
Не всегда возможно, но если есть возможность кормить батчи фиксированного размера, это может положительно сказаться на скорости обучения.

Если размер батчей будет меняться - преаллоцируй память, под батч самого большого размера, чтобы сэкономить на аллокациях.
Преаллокация делается [вот так](https://fastai1.fast.ai/performance.html#faster-image-processing).

### cudnn.benchmark
В недрах `CUDNN` живут разные способы выполнения математических операций.
В зависимости от железа, драйверов и данных они могут показывать разную эффективность.

Если выставить флаг
```
torch.backends.cudnn.benchmark = True
```
на старте тренировки, Torch попробует разные операции. И дальше будет использовать самые эффективные.

При фиксированном размере батча это может ускорить тренировку на несколько процентов.

Если размер батча постоянно меняется - лучше отключить. Скорее всего выбранные операции окажутся неоптимальными и учиться будет очень медленно.

### AdaptiveMixedPrecision или утилизация тензорных ядер
`AdaptiveMixedPrecision` позволяет выполнять часть операций в `16bit`.
Самый эффективный рецепт.

Проверяй что тренировка не взрывается. Внимательно читай документацию, особенно если используешь несколько лосс-функций или занимаешься любмым другим нетривиальным колдунством.

### Apex
[Apex](https://nvidia.github.io/apex/index.html) - расширение для PyTorch от NVidia.

Внутри можно найти реализацию [FusedAdam](https://nvidia.github.io/apex/optimizers.html). Ведет себя как обычный `adam`, но может работать быстрее.

Кроме того, в последних версиях торча появился [параметр](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) `Adam(fused=False)`, можно попробовать включить и померить. Но скорее всего версия из `apex` окажется быстрее.

### jit.Fuse
`Jit.Fuse` [позволяет](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations) склеить несколько операций в одну.

Можно пробовать, если организация кода и операции позволяют.

### Гайды по тюнингу производительности
- [PyTorch performance tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Lightning Speed Up Model Training](https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#set-grads-to-none)
- [Lightning Effective Training Techiques](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html)
- [Huggingface Effective Training On Single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [NVIDIA A100 Application Performance Guide](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/v100-application-performance-guide.pdf)
- [NVIDIA Typical Tile Dimensions In cuBLAS And Performance](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/v100-application-performance-guide.pdf)
- [HSE/YSDA Effective DL](https://github.com/mryab/efficient-dl-systems/tree/main/)

