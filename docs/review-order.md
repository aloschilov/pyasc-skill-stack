# Порядок ручной проверки

Проверяйте происхождение исходников, окружение, измеренные результаты и
содержание skills отдельно. Полный upstream-набор и одна проверка ядра имеют
разный объём покрытия; не подменяйте одно другим.

## Исходники и офлайн-проверки

Используйте чистый checkout или отдельный worktree. Запишите ревизию и состояние
рабочего дерева; журнал и результаты храните вне checkout.

```bash
git rev-parse HEAD
git status --short
bash tests/ci-gate.sh --tier pr
bash scripts/check-target-pin.sh
```

PR gate проверяет skills, инструменты и постоянные regression fixtures.
Происхождение upstream-тестов описано в `reference/asctile/PIN.json` и
[reference/README.md](../reference/README.md). Установка skills и флаг установщика
`--agents` не создают команду агентов; [фазы workflow](ci-gates.md#agent-workflows)
нужно организовать отдельно.

## Закреплённый runtime

Для проверки репозитория используйте development image из текущего пина:

```bash
IMAGE=$(jq -r .dev_image.ref reference/images/ENV.json)
docker pull "$IMAGE"
tests/tools/verify_pyasc_image.sh "$IMAGE"
docker run --rm --network none --ulimit nofile=65536:65536 \
  -v "$PWD":/workspace:ro -w /workspace \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYASC_CACHE_DIR=/tmp/cache -e PYASC_DUMP_PATH=/tmp/dumps \
  "$IMAGE" python3.11 tests/tools/run_target_test.py \
  reference/asctile/target/test_vadd.py \
  --backend Model --platform Ascend950PR_9599
```

По умолчанию helper проверяет компиляцию. Для численного исполнения добавьте
`--run`; для всех случаев модуля — `--full`. Если декларативный список позволяет
выбрать представительный случай, без `--full` исполняется только выбранное
подмножество. Запишите фактически выбранные случаи. Exit 2 означает blocked,
exit 1 — failed; оба результата не дают полного pass. Ни один Model-прогон не
доказывает NPU-исполнение. Для собственного host runtime используйте
[руководство окружения](cann-setup.md), а не произвольное обновление wheel.

## Проверка ядра и skills

Начните с `pyasc-codegen-workflow`, затем загрузите нужные API/синтаксические
skills, `pyasc-code-review` и `pyasc-build-run-verify`. Проверьте наличие дизайна,
ревью с идентификатором сессии, неизменного candidate hash и отчётов проверок.
Исправленный кандидат требует повторного ревью и проверки; сохраняйте оригинал.

Матрица verification contract и её ограничения описаны в
[CI gates](ci-gates.md) и [verification contract](verification-contract.md).
Для hardware и performance используйте официальный контракт CANNBench и
[процедуру отправки](../integrations/cannbench/README.md). Сопоставляйте конкретный
кандидат, runtime и случаи с измерениями; не считайте skill-подсказку ускорением.

Перед принятием результата сверяйте [текущее покрытие](verification-status.md),
CI на проверяемом head и долговременные артефакты. Проверки, не выполненные для
этой ревизии, остаются not-run; исторический успешный прогон их не заменяет.
