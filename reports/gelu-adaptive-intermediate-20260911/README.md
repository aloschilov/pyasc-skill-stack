# GeLU — промежуточный отчёт: адаптация к runner и JIT

Срез: **2026-09-11T05:40:51.473864+00:00**. Кампания 10–11 сентября 2026 остановлена по запросу пользователя.
Таймер удалён; локальных процессов кампании и активных CANNBench jobs на момент
проверки нет. Pending-слоты сохранены, но не будут отправляться автоматически.
Старый отдельный таймер RMSNorm уже имел статус PAUSED и оставлен без изменений.

## Основной результат

**Цель 20/20 и GM>1× не достигнута.** Два успешных адаптивных T дали 20/20,
GM **0.429502×** и **0.447789×**. В обоих 19 из 20 cases медленнее reference.
Контроли A: **0.441481×** и **0.443313×**. Надёжного преимущества только
адаптации запуска эти наблюдения не показали.

В локальном CaModel найдены улучшения FP32 exact и FP16/BF16 exact, но они
**ещё не собраны в квалифицированного финалиста и не измерены на CANNBench**.
FP32 tanh JIT-поиск не начат; следующий поиск геометрии и её взаимодействий
с JIT не выполнен. Отчёт промежуточный, а не заявление о завершении плана.

Использовано **8 из 12 разрешённых попыток**, включая отклонённый duplicate POST;
создано **7 аппаратных jobs**. Доступно **5 credits**;
восстановление по API: `2026-09-11T16:00:00Z`. Credits и лимит 12 попыток — разные
ограничения. Прерванная схема A–B–B–A превратилась в A–B–R–S–T–reject–T–A
из-за ошибок интеграции; исходный чистый A/B-дизайн не был сохранён.

## Что оценивалось и откуда данные

Оператор **только GeLU**,20 официальных cases, FP16/BF16/FP32 × exact/tanh.
Pyasc v2 зафиксирован на `adadd7d66ed0ee16d33d79487bf584899a26ef1e`.
Pass'ы не изменялись. Высокоуровневый AscTile; разрешённое исключение —
inline AscendC `Erfc<float,false>` для FP32 exact. Lowp exact использует
public Erf с FP32-промежуточными, tanh — стандартную cubic exp/sigmoid-формулу.

Архивы CANNBench содержали исполняемый код/совместимый runtime, а не вызов
модели-генератора. Эти результаты оценивают **данные реализации и упаковку**,
не доказывают качество модели или факт генерации с навыками. Основной автор
кандидатов/решений — Codex; OpenCode/Qwen использовался для независимого review
с явно зафиксированной доставкой skill-текста, а не как аппаратный evaluator.

[Данные и SHA256 исходных evidence-файлов](evidence.json) позволяют проверить
таблицы. GM заново рассчитан как `exp(sum(log(reference_us/elapsed_us))/20)`
только при 20 корректных измеренных cases; failed cases не исключаются ради GM.
Ссылки на jobs могут потребовать входа: это **private submissions**.

Снимки исходников: [A](kernels/A.py), [B](kernels/B.py),
[R/S/T — один kernel source](kernels/T.py), [локальный tuning](kernels/local.py).
Это навигационные снимки модулей, не самостоятельные runnable-пакеты:
host helpers, SDK и evaluator wheels намеренно не включены в публикацию отчёта.
Совпадение kernel SHA с ledger проверено для каждого job.

## Все попытки

| Слот | Вариант | Job | Статус | Accuracy | GM× | Runner |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A | [job_86bac3de2260](https://cannbench.com/workspace/jobs/job_86bac3de2260) | succeeded | 20/20 | 0.441481 | runner-950pr-host2-0 |
| 2 | B | [job_27822a212a86](https://cannbench.com/workspace/jobs/job_27822a212a86) | correctness_failed | 17/20 | — | runner-950pr-host2-0 |
| 3 | R | [job_117a8d644d06](https://cannbench.com/workspace/jobs/job_117a8d644d06) | compile_failed | не измерено | — | runner-950pr-host2-0 |
| 4 | S | [job_05b60bc8f9a0](https://cannbench.com/workspace/jobs/job_05b60bc8f9a0) | compile_failed | не измерено | — | runner-950pr-host2-1 |
| 5 | T | [job_9fe60347b927](https://cannbench.com/workspace/jobs/job_9fe60347b927) | succeeded | 20/20 | 0.429502 | runner-950pr-host2-0 |
| 6 | T | — | rejected | не измерено | — | — |
| 7 | T | [job_c22d3768b5a4](https://cannbench.com/workspace/jobs/job_c22d3768b5a4) | succeeded | 20/20 | 0.447789 | runner-950pr-host2-1 |
| 8 | A | [job_8506b5312ebe](https://cannbench.com/workspace/jobs/job_8506b5312ebe) | succeeded | 20/20 | 0.443313 | runner-950pr-host2-1 |
| 9 | F | — | pending | не измерено | — | — |
| 10 | F | — | pending | не измерено | — | — |
| 11 | A | — | pending | не измерено | — | — |

- **1.** Замороженный baseline: прежняя математика, геометрия и фиксированный host cap72.
- **2.** Адаптивный запуск. Cases2/5/8 остановлены host UB preflight: SIMT UB ошибочно принят за AscendC SIMD UB.
- **3.** Попытка исправить UB query. До компиляции ядра сработала проверка checksum внешнего build.sh.
- **4.** Исправлена упаковка. До компиляции ядра выявлено неверное предположение о каталоге metadata SDK.
- **5.** SDK metadata разрешается относительно реального libplatform.so. Первый успешный адаптивный T.
- **6.** Повторная загрузка идентичного ZIP отклонена HTTP400; нового job и списания credit не было. Попытка учтена в лимите 12.
- **7.** Официальный standard reevaluate T; новый submission ID, идентичный SHA256 скачанного архива.
- **8.** Контрольный standard reevaluate исходного A после двух T.
- **9.** Не отправлен: финалист F не квалифицирован; остановлено по просьбе пользователя.
- **10.** Повтор F не отправлен.
- **11.** Финальный контроль A не отправлен.

## Навигация по 20 cases и геометрия аппаратных вариантов

Геометрия ниже вычислена из зафиксированных host selectors A/B/T, не угадана
по имени runner. `A blocks/useful/maxTiles` — **реконструкция из исходника**,
не снятая на устройстве telemetry. Для B/T фактически запрошенные vector cores,
resource limits и launched blocks в сохранённом evidence не подтверждены:
**unknown**. Их нельзя заменять числом 64/72 по суффиксу SoC.

Для всех remote вариантов requested JIT: reuse_alloc=1,static_alloc=None,
insert_sync=True,opt_level=3,debug=False; VF указан по case. Effective JIT на
каждом hardware job отдельно не зафиксирован. В локальном проверенном runtime
None соответствует платформенной static allocation; requested/effective
compiler options проверяются в локальном JIT-поиске.

UB ниже — **локальный compiled UB balanced-ядра при той же геометрии/JIT**,
не аппаратный peak/occupancy и не отдельное измерение A. Подтверждённый SDK
профиль CaModel:253952 байта UB и 72vector cores; это не live query runner.
CaModel timing для каждой полной официальной shape не получен: reduced-N
замеры следующего раздела нельзя подставлять в эти строки.

| Case | Shape | dtype | Mode | tile_shape | unroll | VF | A blocks/useful/maxTiles | Локальный UB,байт |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024×1024 | float16 | none | [8192] | 2 | False | 72/64/2 | 98304 |
| 2 | 2048×2048 | float32 | none | [20480] | 1 | True | 72/69/3 | 245760 |
| 3 | 4096×4096 | bfloat16 | none | [8192] | 2 | False | 72/71/29 | 98304 |
| 4 | 8192×8192 | float16 | tanh | [8192] | 2 | False | 72/72/114 | 131072 |
| 5 | 8192×8192 | float32 | tanh | [15872] | 2 | False | 72/72/59 | 253952 |
| 6 | 1023×1023 | bfloat16 | tanh | [8192] | 2 | False | 72/64/2 | 131072 |
| 7 | 1009×1021 | float16 | none | [8192] | 2 | False | 72/63/2 | 98304 |
| 8 | 1537×769 | float32 | tanh | [15872] | 2 | False | 72/38/2 | 253952 |
| 9 | 363×367×373 | bfloat16 | none | [8192] | 2 | False | 72/72/85 | 98304 |
| 10 | 2049×513 | float16 | tanh | [8192] | 2 | False | 72/65/2 | 131072 |
| 11 | 3×7×13×4001 | float32 | none | [5120] | 1 | True | 72/72/3 | 61440 |
| 12 | 1000003 | bfloat16 | tanh | [8192] | 2 | False | 72/62/2 | 131072 |
| 13 | 11×13×17×67×67 | float32 | none | [5120] | 1 | True | 72/72/30 | 61440 |
| 14 | 3×7×11×13×1009 | float16 | tanh | [8192] | 2 | False | 72/62/6 | 131072 |
| 15 | 512×2049 | float32 | none | [5120] | 1 | True | 72/69/3 | 61440 |
| 16 | 255×8193 | bfloat16 | none | [8192] | 2 | False | 72/64/4 | 98304 |
| 17 | 4097×511 | float16 | tanh | [8192] | 2 | False | 72/64/4 | 131072 |
| 18 | 2×511×2049 | float32 | none | [5120] | 1 | True | 72/69/6 | 61440 |
| 19 | 4×255×2049 | bfloat16 | tanh | [8192] | 2 | False | 72/64/4 | 131072 |
| 20 | 2×3×17×1024×101 | float32 | none | [5120] | 1 | True | 72/72/29 | 61440 |

## Слот 1: A — [job](https://cannbench.com/workspace/jobs/job_86bac3de2260)

Замороженный baseline: прежняя математика, геометрия и фиксированный host cap72.

[Исходник](kernels/A.py); SHA256 `c61785497315aa2b70cf5a20f8808beadc61e151d4b4c3ea87e6b403781a87a5`.

Accuracy **20/20**, anti-cheat failures **0**, GM **0.441481×**.

| Case | Shape | dtype | Mode | Tile/unroll | Accuracy | Reference,µs | Kernel,µs | Speedup× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024×1024 | float16 | none | [8192]/u2 | pass | 4.490 | 11.970 | 0.375 |
| 2 | 2048×2048 | float32 | none | [20480]/u1 | pass | 15.370 | 55.640 | 0.276 |
| 3 | 4096×4096 | bfloat16 | none | [8192]/u2 | pass | 30.140 | 148.010 | 0.204 |
| 4 | 8192×8192 | float16 | tanh | [8192]/u2 | pass | 172.930 | 204.910 | 0.844 |
| 5 | 8192×8192 | float32 | tanh | [15872]/u2 | pass | 387.265 | 337.600 | 1.147 |
| 6 | 1023×1023 | bfloat16 | tanh | [8192]/u2 | pass | 4.460 | 5.800 | 0.769 |
| 7 | 1009×1021 | float16 | none | [8192]/u2 | pass | 4.450 | 12.000 | 0.371 |
| 8 | 1537×769 | float32 | tanh | [15872]/u2 | pass | 6.110 | 8.950 | 0.683 |
| 9 | 363×367×373 | bfloat16 | none | [8192]/u2 | pass | 117.530 | 433.510 | 0.271 |
| 10 | 2049×513 | float16 | tanh | [8192]/u2 | pass | 4.560 | 5.780 | 0.789 |
| 11 | 3×7×13×4001 | float32 | none | [5120]/u1 | pass | 6.050 | 18.800 | 0.322 |
| 12 | 1000003 | bfloat16 | tanh | [8192]/u2 | pass | 4.380 | 5.860 | 0.747 |
| 13 | 11×13×17×67×67 | float32 | none | [5120]/u1 | pass | 37.455 | 160.260 | 0.234 |
| 14 | 3×7×11×13×1009 | float16 | tanh | [8192]/u2 | pass | 7.780 | 11.870 | 0.655 |
| 15 | 512×2049 | float32 | none | [5120]/u1 | pass | 5.980 | 18.470 | 0.324 |
| 16 | 255×8193 | bfloat16 | none | [8192]/u2 | pass | 6.230 | 22.230 | 0.280 |
| 17 | 4097×511 | float16 | tanh | [8192]/u2 | pass | 6.310 | 9.280 | 0.680 |
| 18 | 2×511×2049 | float32 | none | [5120]/u1 | pass | 8.860 | 34.220 | 0.259 |
| 19 | 4×255×2049 | bfloat16 | tanh | [8192]/u2 | pass | 6.260 | 9.230 | 0.678 |
| 20 | 2×3×17×1024×101 | float32 | none | [5120]/u1 | pass | 36.195 | 155.340 | 0.233 |

## Слот 2: B — [job](https://cannbench.com/workspace/jobs/job_27822a212a86)

Адаптивный запуск. Cases2/5/8 остановлены host UB preflight: SIMT UB ошибочно принят за AscendC SIMD UB.

[Исходник](kernels/B.py); SHA256 `4cc38aa30d17f426038f5ec559d511b9fc083a76d2116c901bcdedcdf723fe08`.

Accuracy **17/20**, anti-cheat failures **0**, GM **—×**.

| Case | Shape | dtype | Mode | Tile/unroll | Accuracy | Reference,µs | Kernel,µs | Speedup× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024×1024 | float16 | none | [8192]/u2 | pass | 4.490 | 12.500 | 0.359 |
| 2 | 2048×2048 | float32 | none | [20480]/u1 | fail | 15.370 | — | — |
| 3 | 4096×4096 | bfloat16 | none | [8192]/u2 | pass | 30.140 | 160.720 | 0.188 |
| 4 | 8192×8192 | float16 | tanh | [8192]/u2 | pass | 172.930 | 230.060 | 0.752 |
| 5 | 8192×8192 | float32 | tanh | [15872]/u2 | fail | 387.265 | — | — |
| 6 | 1023×1023 | bfloat16 | tanh | [8192]/u2 | pass | 4.460 | 6.900 | 0.646 |
| 7 | 1009×1021 | float16 | none | [8192]/u2 | pass | 4.450 | 8.130 | 0.547 |
| 8 | 1537×769 | float32 | tanh | [15872]/u2 | fail | 6.110 | — | — |
| 9 | 363×367×373 | bfloat16 | none | [8192]/u2 | pass | 117.530 | 479.020 | 0.245 |
| 10 | 2049×513 | float16 | tanh | [8192]/u2 | pass | 4.560 | 6.940 | 0.657 |
| 11 | 3×7×13×4001 | float32 | none | [5120]/u1 | pass | 6.050 | 18.940 | 0.319 |
| 12 | 1000003 | bfloat16 | tanh | [8192]/u2 | pass | 4.380 | 5.660 | 0.774 |
| 13 | 11×13×17×67×67 | float32 | none | [5120]/u1 | pass | 37.455 | 174.620 | 0.214 |
| 14 | 3×7×11×13×1009 | float16 | tanh | [8192]/u2 | pass | 7.780 | 11.280 | 0.690 |
| 15 | 512×2049 | float32 | none | [5120]/u1 | pass | 5.980 | 18.730 | 0.319 |
| 16 | 255×8193 | bfloat16 | none | [8192]/u2 | pass | 6.230 | 22.770 | 0.274 |
| 17 | 4097×511 | float16 | tanh | [8192]/u2 | pass | 6.310 | 9.970 | 0.633 |
| 18 | 2×511×2049 | float32 | none | [5120]/u1 | pass | 8.860 | 34.420 | 0.257 |
| 19 | 4×255×2049 | bfloat16 | tanh | [8192]/u2 | pass | 6.260 | 10.030 | 0.624 |
| 20 | 2×3×17×1024×101 | float32 | none | [5120]/u1 | pass | 36.195 | 169.820 | 0.213 |

## Слот 3: R — [job](https://cannbench.com/workspace/jobs/job_117a8d644d06)

Попытка исправить UB query. До компиляции ядра сработала проверка checksum внешнего build.sh.

[Исходник](kernels/T.py); SHA256 `e53ce0b5ee0963678a1be4bfc15faca3b18f076f59c8c432ac274a2d4fb02220`.

До исполнения 20 cases не дошло; kernel accuracy, timings и GM не измерены. Это не 20 численных ошибок.

## Слот 4: S — [job](https://cannbench.com/workspace/jobs/job_05b60bc8f9a0)

Исправлена упаковка. До компиляции ядра выявлено неверное предположение о каталоге metadata SDK.

[Исходник](kernels/T.py); SHA256 `e53ce0b5ee0963678a1be4bfc15faca3b18f076f59c8c432ac274a2d4fb02220`.

До исполнения 20 cases не дошло; kernel accuracy, timings и GM не измерены. Это не 20 численных ошибок.

## Слот 5: T — [job](https://cannbench.com/workspace/jobs/job_9fe60347b927)

SDK metadata разрешается относительно реального libplatform.so. Первый успешный адаптивный T.

[Исходник](kernels/T.py); SHA256 `e53ce0b5ee0963678a1be4bfc15faca3b18f076f59c8c432ac274a2d4fb02220`.

Accuracy **20/20**, anti-cheat failures **0**, GM **0.429502×**.

| Case | Shape | dtype | Mode | Tile/unroll | Accuracy | Reference,µs | Kernel,µs | Speedup× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024×1024 | float16 | none | [8192]/u2 | pass | 4.490 | 12.350 | 0.364 |
| 2 | 2048×2048 | float32 | none | [20480]/u1 | pass | 15.370 | 56.210 | 0.273 |
| 3 | 4096×4096 | bfloat16 | none | [8192]/u2 | pass | 30.140 | 160.560 | 0.188 |
| 4 | 8192×8192 | float16 | tanh | [8192]/u2 | pass | 172.930 | 232.120 | 0.745 |
| 5 | 8192×8192 | float32 | tanh | [15872]/u2 | pass | 387.265 | 358.050 | 1.082 |
| 6 | 1023×1023 | bfloat16 | tanh | [8192]/u2 | pass | 4.460 | 6.950 | 0.642 |
| 7 | 1009×1021 | float16 | none | [8192]/u2 | pass | 4.450 | 8.250 | 0.539 |
| 8 | 1537×769 | float32 | tanh | [15872]/u2 | pass | 6.110 | 7.790 | 0.784 |
| 9 | 363×367×373 | bfloat16 | none | [8192]/u2 | pass | 117.530 | 474.060 | 0.248 |
| 10 | 2049×513 | float16 | tanh | [8192]/u2 | pass | 4.560 | 6.990 | 0.652 |
| 11 | 3×7×13×4001 | float32 | none | [5120]/u1 | pass | 6.050 | 18.560 | 0.326 |
| 12 | 1000003 | bfloat16 | tanh | [8192]/u2 | pass | 4.380 | 5.690 | 0.770 |
| 13 | 11×13×17×67×67 | float32 | none | [5120]/u1 | pass | 37.455 | 175.400 | 0.214 |
| 14 | 3×7×11×13×1009 | float16 | tanh | [8192]/u2 | pass | 7.780 | 11.400 | 0.682 |
| 15 | 512×2049 | float32 | none | [5120]/u1 | pass | 5.980 | 18.760 | 0.319 |
| 16 | 255×8193 | bfloat16 | none | [8192]/u2 | pass | 6.230 | 22.740 | 0.274 |
| 17 | 4097×511 | float16 | tanh | [8192]/u2 | pass | 6.310 | 10.090 | 0.625 |
| 18 | 2×511×2049 | float32 | none | [5120]/u1 | pass | 8.860 | 34.160 | 0.259 |
| 19 | 4×255×2049 | bfloat16 | tanh | [8192]/u2 | pass | 6.260 | 9.970 | 0.628 |
| 20 | 2×3×17×1024×101 | float32 | none | [5120]/u1 | pass | 36.195 | 170.320 | 0.213 |

## Слот 7: T — [job](https://cannbench.com/workspace/jobs/job_c22d3768b5a4)

Официальный standard reevaluate T; новый submission ID, идентичный SHA256 скачанного архива.

[Исходник](kernels/T.py); SHA256 `e53ce0b5ee0963678a1be4bfc15faca3b18f076f59c8c432ac274a2d4fb02220`.

Accuracy **20/20**, anti-cheat failures **0**, GM **0.447789×**.

| Case | Shape | dtype | Mode | Tile/unroll | Accuracy | Reference,µs | Kernel,µs | Speedup× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024×1024 | float16 | none | [8192]/u2 | pass | 4.490 | 11.740 | 0.382 |
| 2 | 2048×2048 | float32 | none | [20480]/u1 | pass | 15.370 | 53.580 | 0.287 |
| 3 | 4096×4096 | bfloat16 | none | [8192]/u2 | pass | 30.140 | 160.370 | 0.188 |
| 4 | 8192×8192 | float16 | tanh | [8192]/u2 | pass | 172.930 | 232.510 | 0.744 |
| 5 | 8192×8192 | float32 | tanh | [15872]/u2 | pass | 387.265 | 352.860 | 1.098 |
| 6 | 1023×1023 | bfloat16 | tanh | [8192]/u2 | pass | 4.460 | 5.590 | 0.798 |
| 7 | 1009×1021 | float16 | none | [8192]/u2 | pass | 4.450 | 11.680 | 0.381 |
| 8 | 1537×769 | float32 | tanh | [15872]/u2 | pass | 6.110 | 6.430 | 0.950 |
| 9 | 363×367×373 | bfloat16 | none | [8192]/u2 | pass | 117.530 | 478.930 | 0.245 |
| 10 | 2049×513 | float16 | tanh | [8192]/u2 | pass | 4.560 | 5.710 | 0.799 |
| 11 | 3×7×13×4001 | float32 | none | [5120]/u1 | pass | 6.050 | 17.610 | 0.344 |
| 12 | 1000003 | bfloat16 | tanh | [8192]/u2 | pass | 4.380 | 5.580 | 0.785 |
| 13 | 11×13×17×67×67 | float32 | none | [5120]/u1 | pass | 37.455 | 173.890 | 0.215 |
| 14 | 3×7×11×13×1009 | float16 | tanh | [8192]/u2 | pass | 7.780 | 11.610 | 0.670 |
| 15 | 512×2049 | float32 | none | [5120]/u1 | pass | 5.980 | 17.790 | 0.336 |
| 16 | 255×8193 | bfloat16 | none | [8192]/u2 | pass | 6.230 | 21.910 | 0.284 |
| 17 | 4097×511 | float16 | tanh | [8192]/u2 | pass | 6.310 | 8.660 | 0.729 |
| 18 | 2×511×2049 | float32 | none | [5120]/u1 | pass | 8.860 | 33.700 | 0.263 |
| 19 | 4×255×2049 | bfloat16 | tanh | [8192]/u2 | pass | 6.260 | 8.660 | 0.723 |
| 20 | 2×3×17×1024×101 | float32 | none | [5120]/u1 | pass | 36.195 | 169.090 | 0.214 |

## Слот 8: A — [job](https://cannbench.com/workspace/jobs/job_8506b5312ebe)

Контрольный standard reevaluate исходного A после двух T.

[Исходник](kernels/A.py); SHA256 `c61785497315aa2b70cf5a20f8808beadc61e151d4b4c3ea87e6b403781a87a5`.

Accuracy **20/20**, anti-cheat failures **0**, GM **0.443313×**.

| Case | Shape | dtype | Mode | Tile/unroll | Accuracy | Reference,µs | Kernel,µs | Speedup× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024×1024 | float16 | none | [8192]/u2 | pass | 4.490 | 12.180 | 0.369 |
| 2 | 2048×2048 | float32 | none | [20480]/u1 | pass | 15.370 | 54.500 | 0.282 |
| 3 | 4096×4096 | bfloat16 | none | [8192]/u2 | pass | 30.140 | 148.310 | 0.203 |
| 4 | 8192×8192 | float16 | tanh | [8192]/u2 | pass | 172.930 | 202.400 | 0.854 |
| 5 | 8192×8192 | float32 | tanh | [15872]/u2 | pass | 387.265 | 338.270 | 1.145 |
| 6 | 1023×1023 | bfloat16 | tanh | [8192]/u2 | pass | 4.460 | 5.880 | 0.759 |
| 7 | 1009×1021 | float16 | none | [8192]/u2 | pass | 4.450 | 11.960 | 0.372 |
| 8 | 1537×769 | float32 | tanh | [15872]/u2 | pass | 6.110 | 8.580 | 0.712 |
| 9 | 363×367×373 | bfloat16 | none | [8192]/u2 | pass | 117.530 | 434.650 | 0.270 |
| 10 | 2049×513 | float16 | tanh | [8192]/u2 | pass | 4.560 | 5.860 | 0.778 |
| 11 | 3×7×13×4001 | float32 | none | [5120]/u1 | pass | 6.050 | 18.490 | 0.327 |
| 12 | 1000003 | bfloat16 | tanh | [8192]/u2 | pass | 4.380 | 5.870 | 0.746 |
| 13 | 11×13×17×67×67 | float32 | none | [5120]/u1 | pass | 37.455 | 159.640 | 0.235 |
| 14 | 3×7×11×13×1009 | float16 | tanh | [8192]/u2 | pass | 7.780 | 11.850 | 0.657 |
| 15 | 512×2049 | float32 | none | [5120]/u1 | pass | 5.980 | 18.080 | 0.331 |
| 16 | 255×8193 | bfloat16 | none | [8192]/u2 | pass | 6.230 | 22.640 | 0.275 |
| 17 | 4097×511 | float16 | tanh | [8192]/u2 | pass | 6.310 | 9.310 | 0.678 |
| 18 | 2×511×2049 | float32 | none | [5120]/u1 | pass | 8.860 | 33.330 | 0.266 |
| 19 | 4×255×2049 | bfloat16 | tanh | [8192]/u2 | pass | 6.260 | 9.210 | 0.680 |
| 20 | 2×3×17×1024×101 | float32 | none | [5120]/u1 | pass | 36.195 | 154.280 | 0.235 |

## Локальная матрица JIT — отдельно от аппаратных результатов

Все строки с двумя timings: N262144,8 логических блоков, одинаковая математика
внутри dtype/mode, два свежих процесса; в каждом warmup и 3timed repeats.
Единица — **CaModel ticks**, меньше лучше. Compilation/diagnostic overhead
не входит в ticks. Accuracy охватывает reduced stress, численные границы,
длинный хвост, idle-tail, guards и повторное исполнение, не все 20 полных shapes.
Проверено insert_sync=True,verify_sync=True,opt_level=3,debug=False.
Во всех приведённых представителях static_alloc=None; другие исходные
варианты False/True присутствуют в compile screen, а не проигнорированы.

| Маршрут | Tile/unroll | reuse_alloc | VF | UB,байт | ticks:повтор 1/2 | Статус |
| --- | --- | --- | --- | --- | --- | --- |
| bfloat16/none | [8192]/u2 | 1 | False | 98304 | 21837/21837 | local accuracy+timing pass |
| bfloat16/none | [8192]/u2 | 1 | True | — | — | timeout900 s; accuracy не установлена |
| bfloat16/none | [8192]/u2 | 2 | False | 131072 | 16853/16852 | local accuracy+timing pass |
| bfloat16/none | [8192]/u2 | 2 | True | 131072 | 16631/16631 | local accuracy+timing pass |
| bfloat16/tanh | [8192]/u2 | 1 | False | 131072 | 9636/9636 | local accuracy+timing pass |
| bfloat16/tanh | [8192]/u2 | 1 | True | — | — | timeout900 s; accuracy не установлена |
| bfloat16/tanh | [8192]/u2 | 2 | False | 98304 | 10723/10723 | local accuracy+timing pass |
| bfloat16/tanh | [8192]/u2 | 2 | True | — | — | timeout900 s; accuracy не установлена |
| float16/none | [8192]/u2 | 1 | False | 98304 | 21837/21838 | local accuracy+timing pass |
| float16/none | [8192]/u2 | 1 | True | — | — | timeout900 s; accuracy не установлена |
| float16/none | [8192]/u2 | 2 | False | 131072 | 16853/16852 | local accuracy+timing pass |
| float16/none | [8192]/u2 | 2 | True | 131072 | 16631/16631 | local accuracy+timing pass |
| float16/tanh | [8192]/u2 | 1 | False | 131072 | 9636/9636 | local accuracy+timing pass |
| float16/tanh | [8192]/u2 | 1 | True | — | — | timeout900 s; accuracy не установлена |
| float16/tanh | [8192]/u2 | 2 | False | 98304 | 10723/10723 | local accuracy+timing pass |
| float16/tanh | [8192]/u2 | 2 | True | — | — | timeout900 s; accuracy не установлена |
| float32/none | [5120]/u1 | 1 | True | 61440 | 41142/41141 | local accuracy+timing pass |
| float32/none | [5120]/u1 | 0 | False | 102400 | 35754/35753 | local accuracy+timing pass |
| float32/none | [5120]/u1 | 0 | True | 102400 | 34915/34915 | local accuracy+timing pass |
| float32/none | [5120]/u1 | 2 | False | 61440 | 41307/41307 | local accuracy+timing pass |
| float32/none | [5120]/u1 | 2 | True | 61440 | 40157/40156 | local accuracy+timing pass |
| float32/tanh | [15872]/u2 | 1 | False | — | — | не запускался |
| float32/tanh | [15872]/u2 | 1 | True | — | — | не запускался |
| float32/tanh | [15872]/u2 | 2 | False | — | — | не запускался |
| float32/tanh | [15872]/u2 | 2 | True | — | — | не запускался |

Из 25 представителей **15** прошли все локальные
гейты, **6** отклонены, **4**
не запускались. Исходный compile screen:108JIT-клеток,50 допустимых к дальнейшим
проверкам,20UB-overflow,38unknown/failed. Из 50 компиляций 25 эквивалентных
None/True исключены только после совпадения C++,binary и UB. Отдельный
первичный geometry compile screen:96 клеток,93 допустимы,3overflow; это **не**
новый поиск геометрии с top2JIT, который пока не выполнялся.

### Подтверждённые локальные выводы

- **FP32 exact:** reuse0,VFtrue —34915/34915 против 41142/41141 у reuse1,VFtrue:
  примерно 15,1%меньше ticks при росте UB61440→102400. Отключать VF не нужно.
- **FP16/BF16exact:** reuse2,VFfalse —16853/16852 против≈21837 у reuse1,VFfalse:
  ≈22,8%меньше ticks; UB98304→131072. Добавление VF даёт 16631, ещё≈1,3%,
  то есть само по себе не достигает порога 5% для продвижения отдельной идеи.
- **FP16/BF16tanh:** reuse2 экономит UB131072→98304, но замедляет 9636→10723
  ticks (≈11,3%больше). «Больше свободного UB» не равно «быстрее».
- **FP32 exact reuse2,VFtrue:**40157/40156, лишь≈2,4%лучше baseline, хуже reuse0.
  Удаление одного intermediate store само по себе не определяет общий speedup.

## Lowering, конкуренты и ограничения доказательств

В matched FP32 exact IR прослежено: FindVFGroup объявляет промежуточное x*0.5
выходом группы; LowerToReg материализует transfer; EliminateDataTransfer
внутри DispatchHoist убирает повторную загрузку, но сохраняет store отдельного
выхода. reuse2 делает его overwritten alias и удаляет store без compiler patch.
Несмотря на это, reuse0 быстрее в локальных замерах. Конкретная причина всей
разницы по критическому пути ещё не доказана; minimal standalone reproducer
и расширенное покрытие aliases/branches не завершены. Whole-program sync —
**unknown**, а не pass на основании одного compiler verifier.

EasyAsc/CANNBot Skills (`502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea`) и официальный CANN PyPTO
(`b91b3c7a47823c69382c8be6029280899c05b032`) исследованы в изолированных каталогах.
PyPTO: сборка/import и GeLU FP32[32,128] tensor/instruction lowering получены,
tile[32,32], emitted allocation до 28800 байт; финальный AIV binary/accuracy/perf
не подтверждены. EasyAsc: локальные compile-only артефакты, ограничения
lowp widening/profile. Это **сравнение доступных DSL, не воспроизведение
leaderboard-submissions**; исходники конкретных лидерских submissions не были
подтверждены. Не заявляем преимущество по аппаратной производительности.
Публичные источники: [EasyAsc в CANNBot Skills](https://gitcode.com/cann/cannbot-skills/tree/master/plugins-community),
[официальный CANN PyPTO](https://gitcode.com/cann/pypto),
[CANNBench leaderboard](https://cannbench.com/leaderboard).
В локальных clone manifests зафиксирована лицензия CANN Open Software License
Agreement v2.0. Сборки изолированы; глобальные LLVM/CANN не заменялись.

OpenCode/Qwen reviews сохранены с sessions/hashes; main критически отклонил
неверное замечание об UB-overflow ветке и добавил проверки stale artifacts,
CLI, resume, binary mismatch и overlap. Skill доставлялся полным inline-текстом
в tools-denied reviews, а не через независимый read-tool. Эти review не являются
доказательством использования навыков моделью-генератором каждого kernel.

## Что осталось и почему нет финалиста

1. FP32 tanh:4 локальных JIT-представителя не запущены. Общая матрица не завершена.
2. Top2-per-route поиск tile/unroll/core и повторная JIT-проверка победившей
   геометрии не выполнялись; подготовлены драйверы и 31host-test, не hardware evidence.
3. Старый probe `odd-tail=17*tile+1` содержит 18 тайлов: это long-tail, не odd-count.
   Новый driver задаёт 2*blocks+1 тайлов, но ещё не запускался. Финалист должен
   пройти настоящее odd-count покрытие, полные specialization и evaluator-package
   гейты с совпадением source/wheel hashes.
4. F не собран/не отправлен; слоты 9F,10F,11A остаются неисполненными.
5. Между 03:11UTC и запросом остановки в истории есть поступления heartbeat без
   новых подтверждённых действий агента. Последний запущенный bounded batch
   завершил FP32 exact, но FP32 tanh автоматически не стартовал. Нельзя выдавать
   срабатывание таймера за выполненную работу или утверждать отсутствие простоя.

## Итог

Интеграционные ошибки UB-query/упаковки/SDK-layout исправлены до двух T с 20/20.
Переносимость host launch улучшена, но выигрыш только адаптации не доказан.
Перспективны разные JIT-настройки по маршрутам, а не единый reuse_alloc для всех.
Заполнение UB до предела не цель: данные показывают и выигрыш с большим UB,
и замедление при его экономии. Аппаратный GM остаётся≈0,43–0,45×.
Возобновление — только по новому запросу пользователя; credits автоматически
не расходуются, план не объявлен выполненным.
