# Порядок ручной проверки `pyasc-skill-stack`

Этот документ нужен не для "прогона всего подряд", а для набора доверия к трем вещам:

1. `golden` действительно имеет внятное происхождение.
2. `golden`-ядра действительно исполнимы и корректны.
3. текущий generative-цикл действительно воспроизводит то, что заявлено в `capabilities.yaml` и `evidence/*.json`.

Если какой-то уровень не подтверждается, дальше идти не надо: сначала чинится более фундаментальный уровень.

## Важное предупреждение

Не используйте `~/workspace/README.md` как источник истины для текущего цикла.
Это исторический ручной сценарий, который описывает `@asc.jit`, `asc.data_copy`,
`asc.set_flag`/`asc.wait_flag` и `torch.allclose`, тогда как текущий стек skill-ов
ориентирован на `asc2`, `@asc2.jit(always_compile=True)`, `asc2.load/store`,
`kernel[core_num](...)` и `np.testing.assert_allclose`.

Для текущих ожиданий источником истины являются:

- `skills/pyasc-codegen-workflow/SKILL.md`
- `skills/pyasc-build-run-verify/SKILL.md`
- `skills/pyasc-code-review/SKILL.md`
- `skills/pyasc-api-patterns/SKILL.md`

## 0. Создайте отдельную review-копию

Не проверяйте цикл в текущем рабочем дереве.

Из `~/workspace`:

```bash
cd ~/workspace

git -C pyasc-skill-stack-manual status --short
```

Если `pyasc-skill-stack-manual` чистый, используйте его как review-песочницу.
Если нет, сделайте отдельную копию:

```bash
cd ~/workspace
git clone --shared ./pyasc-skill-stack pyasc-skill-stack-review
cd pyasc-skill-stack-review

git rev-parse HEAD
git status --short
```

Сразу заведите простой журнал наблюдений:

```bash
cat > review-notes.md <<'EOF'
# Review notes

## Provenance

## Golden runtime

## Generative replay

## Decisions
EOF
```

## 1. Сначала проверьте, что заявленное состояние внутренне непротиворечиво

Из review-копии:

```bash
python3.10 tests/tools/check_capabilities.py
bash tests/run-tests.sh --fast
```

Что это дает:

- `check_capabilities.py` проверяет согласованность `capabilities.yaml` с `golden` и `evidence`
- `--fast` прогоняет L1-слой: быстрые детерминированные проверки структуры и содержимого
  `skills/`, `teams/`, `agents/` и согласованности repo-артефактов
- это первый фильтр неопределенности: сначала нужно убедиться, что сам stack
  структурно цел, и только потом разбирать agent behavior, runtime и nightly-нестабильность

Если это уже не проходит, не анализируйте nightly и prompt tuning.
Сначала надо чинить базовую целостность репозитория.

## 2. Подтвердите происхождение `golden/docs` и `golden/tutorials`

Сначала проверьте, что это действительно локальный snapshot из соседнего `pyasc`.

```bash
diff -rq golden/docs ~/workspace/pyasc/docs
```

Если `diff -rq` показывает отличия, разберите их поштучно.
Минимальный набор для ручного просмотра:

```bash
diff -u golden/docs/architecture_introduction.md ~/workspace/pyasc/docs/architecture_introduction.md | sed -n '1,160p'
diff -u golden/docs/python_syntax_support.md ~/workspace/pyasc/docs/python_syntax_support.md | sed -n '1,160p'
diff -u golden/docs/developer_guide.md ~/workspace/pyasc/docs/developer_guide.md | sed -n '1,160p'
```

Затем сравните tutorial snapshot-ы:

```bash
diff -u golden/tutorials/01_add.py ~/workspace/pyasc/python/tutorials/01_add/add.py | sed -n '1,160p'
diff -u golden/tutorials/02_add_framework.py ~/workspace/pyasc/python/tutorials/02_add_framework/add_framework.py | sed -n '1,160p'
diff -u golden/tutorials/03_matmul_mix.py ~/workspace/pyasc/python/tutorials/03_matmul_mix/matmul_mix.py | sed -n '1,160p'
diff -u golden/tutorials/04_matmul_cube_only.py ~/workspace/pyasc/python/tutorials/04_matmul_cube_only/matmul_cube_only.py | sed -n '1,160p'
diff -u golden/tutorials/05_matmul_leakyrelu.py ~/workspace/pyasc/python/tutorials/05_matmul_leakyrelu/matmul_leakyrelu.py | sed -n '1,160p'
```

Если хотите проверить ссылочную часть из `developer_guide.md`, только тогда подключайте соседний `asc-devkit`.
До этого он не нужен.

Решение после этого шага:

- если `golden/docs` и `golden/tutorials` почти совпадают с `~/workspace/pyasc`, их можно считать snapshot-референсом
- если отличия большие и непонятные, сначала фиксируйте provenance, а не skill-ы

## 3. Подтвердите, что `golden/kernels` исполнимы как эталон

Подготовьте окружение CANN:

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
python3.10 -c "import asc, asc2; print('asc/asc2 OK')"
```

Сначала пройдите статическую и JIT-проверку всех golden kernels:

```bash
for k in golden/kernels/*.py; do
  echo "== $k =="
  python3.10 tests/tools/verify_kernel.py "$k" || break
  python3.10 tests/tools/run_and_verify.py "$k" --mode jit || break
done
```

Потом руками прогоните runtime хотя бы по одной ячейке на каждый паттерн:

```bash
python3.10 tests/tools/run_and_verify.py golden/kernels/abs_f16.py --mode simulator
python3.10 tests/tools/run_and_verify.py golden/kernels/reduce_sum_f32.py --mode simulator
python3.10 tests/tools/run_and_verify.py golden/kernels/gelu_f16.py --mode simulator
python3.10 tests/tools/run_and_verify.py golden/kernels/leaky_relu_f16.py --mode simulator
python3.10 tests/tools/run_and_verify.py golden/kernels/softmax_f16.py --mode simulator
```

Если есть силы, отдельно проверьте и `gelu_f32`:

```bash
python3.10 tests/tools/run_and_verify.py golden/kernels/gelu_f32.py --mode simulator
```

Решение после этого шага:

- если golden kernels сами не проходят, нельзя считать skill-ы или evidence надежными
- если golden kernels проходят, можно считать их рабочим эталоном поведения

## 4. Пересоберите golden evidence в dry-run и сравните с checked-in JSON

Не перезаписывайте сразу `evidence/`.
Сначала воспроизведите JSON на экран.

```bash
python3.10 tests/tools/collect_evidence.py golden/kernels/abs_f16.py \
  --op abs --dtype float16 --kind golden \
  --shapes '[[1,128],[4,2048],[32,4096]]' \
  --runtime --dry-run

python3.10 tests/tools/collect_evidence.py golden/kernels/softmax_f16.py \
  --op softmax --dtype float16 --kind golden \
  --shapes '[[32,4096]]' \
  --runtime --dry-run
```

Потом сравните результат глазами с checked-in evidence:

```bash
jq '{operation,dtype,verification,score,static_verify,kernel_path}' evidence/abs-f16-golden.json
jq '{operation,dtype,verification,score,static_verify,kernel_path}' evidence/softmax-f16-golden.json
```

Цель этого шага не в том, чтобы даты совпали.
Цель в том, чтобы совпадали:

- `kernel_path`
- `static_verify`
- `verification.status`
- `score.accepted`
- общая логика прохождения

## 5. Только теперь смотрите на checked-in generative evidence

Сначала посмотрите сводку по верхнему уровню каждого JSON.
Не начинайте с `history`.

```bash
for f in evidence/*-generative.json; do
  echo "== $f =="
  jq '{operation,dtype,date,verification,semantic_check,score,kernel_path}' "$f"
done
```

Отдельно выпишите подозрительные случаи:

```bash
jq '{operation,dtype,date,verification,semantic_check,score,kernel_path,history:(.history|length)}' evidence/gelu-f32-generative.json
jq '{operation,dtype,date,verification,semantic_check,score,kernel_path,history:(.history|length)}' evidence/matmul-f16-generative.json
```

Вопросы, на которые надо ответить руками:

1. runtime действительно `pass` или `fail`?
2. semantic check действительно отражает смысл операции?
3. `kernel_path` похож на правильный артефакт?
4. причина `pending` в модели, в чекере или в сборщике evidence?

## 6. Воспроизведите generative-цикл на трех representative случаях

Берите не все операции, а три:

- `abs/float16` как легкий подтверждающий кейс
- `softmax/float16` как тяжелый, но рабочий кейс
- `gelu/float32` как текущий спорный кейс

Создайте архив для артефактов:

```bash
mkdir -p /tmp/pyasc-gen-review
```

Запускайте по одному:

```bash
python3.10 tests/tools/collect_generative_evidence.py \
  --op abs --dtype float16 \
  --runtime --keep-project \
  --archive-dir /tmp/pyasc-gen-review

python3.10 tests/tools/collect_generative_evidence.py \
  --op softmax --dtype float16 \
  --runtime --keep-project \
  --archive-dir /tmp/pyasc-gen-review

python3.10 tests/tools/collect_generative_evidence.py \
  --op gelu --dtype float32 \
  --runtime --keep-project \
  --archive-dir /tmp/pyasc-gen-review
```

После каждого запуска фиксируйте в `review-notes.md`:

- какой `Project:` был создан
- какой `Kernel:` был найден
- что показали `Static verify`, `Semantic check`, `Runtime`
- совпадает ли найденный kernel с ожидаемой операцией

Если `gelu/float32` снова приводит к чужому `kernel.py`, это уже не "слабый prompt",
а проблема в harness-е поиска артефакта.

## 7. Осмотрите реальные артефакты, а не только JSON

После re-run откройте архив и руками прочитайте результаты:

```bash
find /tmp/pyasc-gen-review -maxdepth 4 -type f | sort | sed -n '1,200p'
```

Для каждого кейса откройте:

- `kernel.py`
- `design.md`
- `self_review.md`
- `acceptance_review.md`
- `verification.md`

Что проверять глазами:

1. операция в `kernel.py` действительно соответствует prompt
2. нет ли v1 API там, где stack уже ожидает asc2
3. verification-код соответствует текущим правилам stack-а
4. self/acceptance review не являются пустым формальным мусором

## 8. Только после этого читайте сами SKILLS

Не читайте весь каталог `skills/` подряд.
Порядок для загрузки контекста:

1. `skills/pyasc-api-patterns/SKILL.md`
2. `skills/pyasc-build-run-verify/SKILL.md`
3. `skills/pyasc-codegen-workflow/SKILL.md`
4. `skills/pyasc-docs-search/SKILL.md`
5. затем только нужные вам специализированные skill-ы

Смысл такой:

- сначала понять канонический кодовый паттерн
- потом понять канонический verification pattern
- потом понять workflow exit criteria
- потом понять, на каких источниках документации это все основано

## 9. После проверки принимайте решение по такому правилу

Если ломается provenance `golden/docs` или `golden/tutorials`, сначала чините provenance snapshot-ов.

Если `golden/kernels` не проходят руками, сначала чините golden set, а не prompt engineering.

Если golden проходит, а generative-цикл нестабилен, улучшайте prompts, workflow и review-этапы.

Если runtime проходит, но status остается `pending`, сначала проверяйте:

- semantic markers
- `find_kernel()` в `tests/tools/collect_generative_evidence.py`
- логику `sync_capabilities.py`

Если `kernel_path` указывает не на тот артефакт, не принимайте решения по качеству skill-ов,
пока не починен evidence harness.

## Короткая версия

Идти нужно в таком порядке:

1. clean review copy
2. `check_capabilities.py` и `--fast`
3. provenance `golden/docs` и `golden/tutorials`
4. runtime golden kernels
5. dry-run golden evidence
6. audit checked-in generative evidence
7. replay `abs`, `softmax`, `gelu/f32`
8. inspection archived artifacts
9. reading core skills
10. только потом решения по улучшениям
