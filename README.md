#  Как учить Deep Learning модели быстрее

Советы по ускорению тренировок в рамках одной GPU.
Как максимально быстро кормить данные в GPU.

0. Бенчмаркай
Перед тем как начать что-то улучшать, научись это измерять.
Используй [профайлеры](https://pytorch.org/tutorials/recipes/recipes/benchmark.html).
Проверяй что тренировка не разваливается, после внесения изменений.

1. Достань более быструю GPU
Каким бы образом ты не извращался, допрыгнуть до следующей GPU в модельном ряду, скорее всего, не получится.
Поэтому, если есть возможность достать GPU - доставай.

2. Проверь боттлнеки на чтение данных

2.1 DataLoader(num_workers=?)
По умолчанию DataLoader будет готовить данные для обучения в основном потоке исполнения.
Попробуй выставить `num_workers=1`, чтобы отселить загрузку данны в отдельные процесс.
С помощью профайлера и бенчмарков определи оптимальное число воркеров. 

2.2 pin_memory + non_blocking
Параметр `DataLoader(pin_memory=True)` в связке с `Tensor.to(device, non_blocking=True)`, может ускорить загрузку тензоров на GPU.

2.1 Используй более быстрые ридеры
Для картинок:

3. Убирай синхронизацию
Любые вызовы `Tensor.cpu()/to()/numpy()` заставляют GPU синхронизироваться. Синхронизация это дорого.
Выкидывай все ненужные синхронизации.

4. Увеличивай батч-сайз

5. Фиксированный размер батч-сайза
  - тулзы для паковки батч-сайза

6. cudnn.benchmark
В недрах CUDNN живут разные способы вычисления математических операций.
В зависимости от железа, драйверов и данных они могут показывать разную эффективность.
При выставлении флага
```
torch.backends.cudnn.benchmark = True
```
на старте тренировки Torch попробует разные операции и дальше будет использовать самые эффективные.
При фиксированном размере батча может ускорить тренировку на несколько процентов.
Если размер батча постоянно меняется - лучше отключить. Скорее всего выбранные операции окажутся неоптимальными и учиться будет очень медленно.

7. AdaptiveMixedPrecision или утилизация тензорных ядер
AdaptiveMixedPrecision позволяет выполнять часть операций в 16bit.
Самый эффективный рецепт улучшения.
Проверяй что тренировка не взрывается. Встречал проблемы с CTC лоссом.

8. Apex
[Apex](https://nvidia.github.io/apex/index.html) - расширение для PyTorch от NVidia.
Внутри можно найти реализацию [FusedAdam](https://nvidia.github.io/apex/optimizers.html). Ведет себя как обычный `adam`, но может работать быстрее.
Кроме того, в последних версиях торча появился [параметр](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) `Adam(fused=False)`, можно попробовать включить и померить. Но скорее всего версия из `apex` окажется быстрее.

9. jit.Fuse
`Jit.Fuse` [позволяет](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations) склеить несколько операций в одну.
Можно пробовать, если организация кода и операции позволяют.

10. Мелкие хаки 
Сбрасывать градиенты вместо `zero_grad`.
Можно выставить `Optimizer.zero_grad(set_to_none=True)` и надеяться на небольшой прирост производительности.
Важно понимать, что такое изменение может изменить ход обучения. Подробнее в [документации](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch-optim-optimizer-zero-grad)

11. Гайды по тюнингу производительности

12. Используй оффициальный докер образ NVidia