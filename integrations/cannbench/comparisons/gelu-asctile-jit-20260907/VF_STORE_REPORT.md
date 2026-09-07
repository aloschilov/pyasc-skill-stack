# Уточнение: лишние VF-store — GeLU / AscTile JIT

Уточнение от 2026-09-07 к обсуждению «Лишние VF-store performance»
(conversation ID `6a9e8737-ee50-83eb-bda1-ef2209fc7c30`). Здесь разделены
численная точность exact FP32 GeLU и качество lowering. Это фиксация
существующих результатов, без нового запуска или изменения compiler passes.

## Correctness: CANN Erf и cancellation

Exact GeLU вычисляется как `z=x/sqrt(2)`, `a=erf(z)`, `b=1+a`,
`y=0.5*x*b`. В отрицательном хвосте `a` близко к -1, поэтому cancellation
в `1+erf(z)` усиливает ошибку backend primitive.

[Численный reproducer](evidence/numeric-reproducer.json) при
`x=-4.111328125`, `z=-2.9071478843688965` показывает CANN Erf
`-0.9999605417251587` вместо корректно округлённого FP32
`-0.9999606609344482`: ошибка 2 ULP. Получается GeLU
`-8.111295755952597e-5` против float64 golden `-8.086769955458579e-5`;
checker MARE = `0.003029084196911871`, cancellation failures = 64/64.
Подстановка корректно округлённого Erf даёт CPU-результат в этой точке
и ноль CPU cancellation failures. Это не доказательство неизбежного
провала любой FP32 реализации.

Reproducer использует 64 одинаковых значения без tiled loop, с
`reuse_alloc=0`, `static_alloc=False`, `vf_fusion=False` и sync verification.
Все [18 exact FP32 JIT settings](evidence/numerical-dense-exact/summary.json)
провалили плотную проверку 4097 точек на [-10,10]. Измерения относятся к
pin `adadd7d66ed0ee16d33d79487bf584899a26ef1e`, локальному CANN 9.0 /
Ascend950PR_9599; они не устанавливают результат на непроверенном remote runtime.

**Причина изолированного correctness failure — точность публичного backend
CANN Erf, усиленная cancellation. Данные не приписывают этот сбой
DispatchHoist, EliminateDataTransfer, VF grouping или reuse_alloc.**
Исправление lowering само по себе не снимает численный gate; нужна повторная
проверка точности поддерживаемой реализации Erf либо устойчивой exact GeLU.

## Performance: что означает «лишние VF-store»

Речь о промежуточной материализации результатов vector-function (VF) в UB.
Для цепочки elementwise-операций желательно передавать промежуточные значения
в регистрах внутри совместимой VF-группы. Цикл
`compute → store UB → load UB → compute` может добавлять UB traffic,
занимать UB и требовать дополнительных зависимостей или синхронизации.
Даже после удаления повторного load может остаться ненужный store.

Это отдельная performance/lowering проблема. Для memory-bound режима такие
передачи потенциально существенны, но само наличие stores в IR не доказывает,
что конкретный запуск GeLU ограничен памятью или что все передачи переживают
Bisheng. Здесь нет измеренного ускорения от их устранения и нет утверждения,
что они объясняют всю разницу с baseline или Model timeouts.

## reuse_alloc, VF fusion и 108 конфигураций

[Compile matrix](evidence/compile-matrix/summary.json) содержит **108/108
успешных compilation + sync verification**: 3 dtype × 2 mode ×
`reuse_alloc={0,1,2}` × `static_alloc={None,False,True}` ×
`vf_fusion={False,True}` при tile1024/unroll1 и одном SHA256 candidate
`82da6f0be5cb38f0cbc4a748250a9d7c2e5b007d876c323a77d16a0f30ead463`.
Это 18 комбинаций флагов на каждый dtype/mode, а не 108 разных наборов флагов.
Compile/sync success не доказывает numerical correctness, полную безопасность
повторных запусков или высокую производительность.

`reuse_alloc` переиспользует UB allocations по временам жизни; это не уровень
register allocation. Он уменьшает UB footprint и меняет идентичность
назначений, что может помочь позднему удалению перезаписанных stores.
VF fusion объединяет совместимые вычисления и позволяет передавать значения
через регистры. Эти механизмы дополняют друг друга, но не гарантируют удаления
всех materializations и не исправляют точность Erf.

[Сохранённый lowering-анализ](evidence/lowering-analysis.json) показывает для
exact с VF fusion: stores внутри хвостовой VF-группы при reuse0/1/2 = 3/2/1,
из них обязательный output store = 1, избыточные = 2/1/0. Для tanh-exp
остаётся соответственно 9/3/1 избыточных stores. Поэтому нельзя говорить,
что каждый из 108 вариантов обязательно содержит лишний exact-tail store.
Reuse1 и reuse2 дают одинаковые 12288 UB bytes для exact FP32, хотя число
stores различается. Уменьшение UB capacity не равнозначно уменьшению traffic
или гарантированному ускорению; VF fusion в имеющихся малых timing controls
не был равномерно быстрее.

## Обязательные stores и направление pass-анализа

Обязательны относительно текущего ABI и границ групп: сохранение итогового
`y` для copy-out/конечного cast, а также передачи на границе LocalTensor CANN
Erf. Выбранный LowerMath ABI материализует вход и выход Erf; у Erf нет
совместимого LowerToReg conversion. Эти передачи нельзя назвать мёртвыми
промежуточными stores. Иной register-capable lowering мог бы изменить границу,
но требует отдельной реализации, а не только добавления Erf в список fusible.

Избыточен store промежуточного значения, если дальнейшие операции уже получают
его регистр и его сохранённое значение никто не читает вне группы, с учётом
aliasing и loop liveness. В exact/reuse0 это `erf_result+1` и
`x*(erf_result+1)`; только последний результат нужен для copy-out.

[Подробный LOWERING](LOWERING.md) уже фиксирует последовательность IR:

1. После VF grouping (`FindVFGroup`) все три arithmetic destinations попадают
   в `dstList`, хотя наружу нужен только конечный результат. Это наблюдаемая
   точка возникновения ложных group outputs.
2. После LowerToReg / FuseVFFor видны четыре loads и три stores.
   Вложенный `EliminateDataTransfer` внутри `DispatchHoist` действительно
   выполняется: первая итерация уменьшает loads до двух, но оставляет три
   stores; вторая сохраняет этот результат. Pass присутствует, а не пропущен.
3. Последние stores к объявленным outputs сохраняются. После
   MaterializeLoadStore в итоговом AscIR остаются две register-load и три
   register-store sites. Это счётчики статических мест, не динамических событий.

Направление исправления — проследить каждый tensor до и после VF grouping,
DispatchHoist и вложенных EliminateDataTransfer; определить фактические внешние
чтения, aliases, перезаписи и время жизни. Затем уточнить `dstList`, дать
существующему pass удалить действительно мёртвые stores, очистить ненужные
allocations и проверить masks, tails, sync и повторные итерации. ABI transfers
классифицируются отдельно. Нужна проверка backend machine code/профилирование,
чтобы установить реальное влияние на время, без переноса этой гипотезы на
численный сбой.

Увеличение tile или изменение числа cores не заменяет этот анализ.
Историческая FP16 exact производительность ниже baseline относилась к другим,
низкоуровневым кандидатам; она не является timing evidence данного high-level
candidate. Наличие 72 vector cores в конфигурации платформы само по себе не
доказывает их эффективную загрузку.

## Сохранность evidence и объём публикации

Вместе с уточнением опубликованы четыре исходных JSON по ссылкам выше,
побайтно без изменений, а также существующие README и LOWERING. Их исходные
ссылки и описания сохранены. Остальные raw logs, generated IR/C++, binaries,
runtime trees, harnesses и worker artifacts остаются в локальном comparison
каталоге; ссылки на эти неопубликованные файлы в исходных документах и пути
в JSON могут быть недоступны в web. Эта публикация не является полным
reproduction bundle и не добавляет новые результаты измерений.
