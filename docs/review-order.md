# Порядок ручной проверки `pyasc-skill-stack`

Документ нужен не для «прогона всего подряд», а для набора доверия по уровням:

1. репозиторий структурно цел;
2. эталон (`reference/asctile/target`) действительно имеет внятное происхождение;
3. эталон действительно исполним на этом хосте;
4. и только потом — содержание самих skill-ов.

Если уровень не подтверждается, дальше идти не надо: сначала чинится более
фундаментальный уровень.

## Что здесь считается эталоном

Только `reference/asctile/target` — набор тестов, написанный Huawei и
вендоренный из upstream `pyasc`. **Ядро, написанное моделью, эталоном не
является** — ни для корректности, ни для тюнинга, ни как база для сравнения
производительности. Подробнее: [`evidence-sources.md`](evidence-sources.md).

Старый каталог `golden/` с ядрами от модели удалён; он доступен из ветки
`archive/campaign-2026`.

## 0. Отдельная review-копия

Не проверяйте в рабочем дереве.

```bash
cd ~/workspace
git clone --shared ./pyasc-skill-stack pyasc-skill-stack-review
cd pyasc-skill-stack-review
git rev-parse HEAD
git status --short
```

Заведите журнал наблюдений `review-notes.md` с разделами: provenance, runtime,
skills, decisions.

## 1. Структурная целостность

```bash
bash tests/run-tests.sh --fast
```

Прогоняет L1: детерминированные проверки структуры и содержимого `skills/`,
`agents/`, согласованности repo-артефактов, плюс офлайн-проверку пина эталона.
Сети не требует.

Если это не проходит — не анализируйте runtime и поведение агента, чините
базовую целостность.

## 2. Происхождение эталона

Пин описан в `reference/asctile/PIN.json`: upstream, путь, коммит и sha256
каждого файла.

```bash
bash scripts/check-target-pin.sh
```

Проверка офлайн. Ненулевой код возврата означает расхождение содержимого с
пином, а не «коммит не разрешён» — поле `commit: unresolved` само по себе не
ошибка, см. раздел «Известные ограничения» в [`../CHANGELOG.md`](../CHANGELOG.md).

Что проверить глазами:

- `reference/README.md` — чем снимок отличается от upstream и почему;
- локальные правки должны быть только в `conftest.py` (алиас импорта) и
  помечены комментарием `Local addition, not upstream`;
- сами `test_*.py` должны быть байт-идентичны upstream, иначе сравнение по
  хешам теряет смысл.

Пересобрать снимок (нужна сеть):

```bash
bash scripts/sync-target-suite.sh --dry-run    # посмотреть, что изменится
bash scripts/sync-target-suite.sh              # применить, затем прочитать диф
```

Скрипт ищет набор **по содержимому** (модули, использующие и `asctile.jit`, и
`asctile.global_tensor`), а не по фиксированному пути: upstream переносит его
от версии к версии (`python/test/asc2/target` → `python/test/asctile/target` →
`experimental/asctile/python/test/target`).

## 3. Эталон исполним

Нужен установленный runtime. Чистый venv с опубликованным колесом:

```bash
python3 -m venv /tmp/target-suite-venv
/tmp/target-suite-venv/bin/pip install --pre pyasc-dev
```

Проверьте, что это именно тот runtime, который вы думаете:

```bash
/tmp/target-suite-venv/bin/python -c \
  "from asc.experimental import asctile; print(asctile.__file__)"
```

Если вместо этого импортируется `asctile` из editable-копии и падает с
`module 'asctile' has no attribute 'jit'` — это пустой namespace package из
устаревшего checkout-а на `sys.path`. `conftest.py` диагностирует такой случай
отдельным понятным сообщением.

Сначала сбор без запуска:

```bash
cd ~/workspace/pyasc-skill-stack-review
/tmp/target-suite-venv/bin/python -m pytest reference/asctile/target --collect-only -q | tail -1
```

Ожидается 1155 тестов. Затем окружение CANN и запуск одного модуля:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"

/tmp/target-suite-venv/bin/python -m pytest reference/asctile/target/test_vadd.py \
  --backend Model --platform Ascend950PR_9599
```

`Backend.Model` — симулятор, не железо. Пройденный тест здесь означает
«прошло на симуляторе», и в отчёте это должно быть написано именно так.

Модули тяжёлые (у каждого ~100 параметризованных случаев), поэтому для
быстрой проверки добавляйте `--compile-only`. Это проверяет генерацию и
компиляцию, но **не** численный результат.

Решение после шага: если эталон не проходит руками, нельзя считать надёжными
ни skill-ы, ни любые отчёты, опирающиеся на них.

## 4. Текущее состояние проверок

Прочитайте [`verification-status.md`](verification-status.md) — там
зафиксировано, какие гейты пройдены, какие `blocked` и почему. Не повторяйте
то, что уже задокументировано как заблокированное на стороне провайдера.

## 5. Только теперь — сами skill-ы

Не читайте каталог `skills/` подряд. Порядок загрузки контекста:

1. `skills/pyasc-api-patterns/SKILL.md` — канонический кодовый паттерн;
2. `skills/pyasc-build-run-verify/SKILL.md` — канонический паттерн верификации;
3. `skills/pyasc-codegen-workflow/SKILL.md` — workflow и exit criteria;
4. `skills/pyasc-docs-search/SKILL.md` — на каких источниках это основано;
5. далее только нужные специализированные skill-ы.

При чтении сверяйте с реальностью: текущий API — `asctile`, импорт
`from asc.experimental import asctile` с запасным вариантом `import asctile`.
Упоминания `asc.data_copy`, `asc.set_flag`/`asc.wait_flag`, `@asc2.jit`,
`asc2.load/store` — признак устаревшего текста.

## 6. Правило принятия решений

- Ломается пин или provenance эталона → чините provenance, а не skill-ы.
- Эталон не исполняется → чините окружение/эталон, а не prompt engineering.
- Эталон проходит, а генеративный цикл нестабилен → улучшайте prompts,
  workflow и review-этапы.
- Нужна цифра производительности → только CANNBench, и только против
  определённого случая TTK; иначе честный `blocked`, а не придуманная база.

## Короткая версия

1. чистая review-копия
2. `tests/run-tests.sh --fast`
3. `scripts/check-target-pin.sh` + чтение `reference/README.md`
4. сбор и запуск эталона на симуляторе
5. `verification-status.md`
6. чтение основных skill-ов
7. и только потом решения по улучшениям
