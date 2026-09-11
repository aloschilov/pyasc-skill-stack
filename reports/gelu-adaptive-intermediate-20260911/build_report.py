"""GET-only report export; no launches, uploads, ledger edits or compiler work.

Requires the task-local evidence checkout. Published JSON contains selected
metrics/source hashes, not credentials, account identifiers or raw service logs.
All report files are generated using apply_patch. Existing outputs are refused.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import sys

import yaml

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
CAMP = REPO / 'integrations/cannbench/comparisons/gelu-adaptive-20260910'
sys.path[:0] = [str(CAMP), str(CAMP / 'queue')]
from controller_v6 import Controller, ReadOnlyClient, snapshot
from collect_job_v6 import summarize
from rank_jit import audit

manifest = {}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    manifest[str(path.relative_to(REPO))] = sha(path)
    return path.read_text()


def publish(path, content):
    if path.exists():
        if '--refresh' not in sys.argv: raise RuntimeError('Use --refresh for report regeneration: ' + str(path))
        before = path.read_text()
        if before == content: return
        patch = '*** Begin Patch\n*** Update File: ' + str(path) + '\n@@\n' + ''.join('-' + s + '\n' for s in before.splitlines()) + ''.join('+' + s + '\n' for s in content.splitlines()) + '*** End Patch\n'
    else:
        patch = '*** Begin Patch\n*** Add File: ' + str(path) + '\n' + ''.join('+' + s + '\n' for s in content.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, check=True, capture_output=True)


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, str) and value.startswith(str(REPO) + '/'):
        return value[len(str(REPO)) + 1:]
    return value


def num(value, places=3): return '—' if value is None else f'{value:.{places}f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def main():
    state = Controller().status()
    observed = snapshot(ReadOnlyClient())
    assert not observed['active_jobs'], 'Unexpected active remote job: review before publishing stop report'
    cases = yaml.safe_load(read(REPO / 'integrations/cannbench/tasks/gelu/cases.yaml'))['cases']
    assert len(cases) == 20 and {c['case_id'] for c in cases} == set(range(1, 21))
    ranking = audit()
    implementations = {'A': 'A', 'B': 'balanced_context', 'T': 'balanced_metadata', 'local': 'balanced'}
    sources = {}
    for label, variant in implementations.items():
        path = CAMP / 'variants' / variant / 'gelu.py'
        source = read(path)
        sources[label] = dict(sha256=sha(path), path=f'kernels/{label}.py')
        publish(OUT / 'kernels' / (label + '.py'), source)
    # Only two known pure host selector functions are evaluated, no kernel imports/launches.
    tree = ast.parse((OUT / 'kernels/A.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('geometry', 'select_tile')]
    assert len(functions) == 2
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'verified-host-selectors', 'exec'), namespace)
    navigation = []
    for c in cases:
        shape, dtype, mode = c['input_shape'][0], c['dtype'][0], c['attrs']['approximate']
        n = math.prod(shape)
        tile, unroll, vf = namespace['geometry'](dtype == 'float32', mode)
        tile = namespace['select_tile'](n, dtype == 'float32', mode, tile)
        blocks = min(72, (n + tile - 1) // tile)
        span = ((n + blocks - 1) // blocks + tile - 1) // tile
        useful = (n + tile * span - 1) // (tile * span)
        key = f'dtype{dtype}-mode{mode}-tile{tile}-unroll{unroll}-reuse1-staticdefault-vf{int(vf)}'
        path = CAMP / 'screen/geometry' / key / 'cell.json'
        measured = json.loads(read(path))
        ub = measured.get('ub') if measured.get('eligible_for_numerical') else None
        navigation.append(dict(case_id=c['case_id'], shape=shape, dtype=dtype, mode=mode,
            tile_shape=[tile], unroll=unroll, vf_fusion=vf, baseline_requested_blocks=blocks,
            baseline_useful_blocks=useful, baseline_max_tiles_per_block=span,
            local_balanced_compile_ub_bytes=ub, remote_reported_aiv=None, remote_limited_aiv=None,
            adaptive_launched_blocks=None))
    runs = []
    for slot in state['slots']:
        row = {k: slot.get(k) for k in ('slot', 'candidate', 'state', 'job_id')}
        if slot.get('job_id'):
            path = CAMP / 'remote' / slot['tag'] / 'job.json'
            payload = json.loads(read(path)); job = payload.get('job', payload)
            row.update(status=job['status'], runner=job.get('runner_id'),
                       started=job.get('claimed_at'), finished=job.get('finished_at'),
                       url='https://cannbench.com/workspace/jobs/' + job['id'],
                       archive_sha256=slot['binding']['identity']['archive_sha256'],
                       source_sha256=slot['binding']['identity']['source_sha256'])
            source_key = 'T' if row['candidate'] in ('R', 'S', 'T') else row['candidate']
            assert sources[source_key]['sha256'] == row['source_sha256']
            row['kernel'] = sources[source_key]['path']
            # Failed package builds may contain 20 service-created failed rows;
            # these are NOT executed numerical tests.
            row['summary'] = None if job['status'] == 'compile_failed' else summarize(job)
            if row['summary'] and row['summary']['gm_all20']:
                assert math.isclose(row['summary']['gm_all20'], math.exp(sum(math.log(x['reference_us'] / x['elapsed_us']) for x in row['summary']['rows']) / 20), rel_tol=1e-12)
        runs.append(row)
    captured = datetime.now(timezone.utc).isoformat()
    credits = {k: observed['credits']['credits'].get(k) for k in ('remaining', 'resets_at', 'timezone')}
    phases = []
    for path in sorted((CAMP / 'tuning').glob('*/*/PROCESS.json')):
        record = json.loads(read(path))
        phases.append(dict(cell=record['identity']['cell'], phase=path.parent.name, passed=record['passed'],
                           exit_code=record['exit_code'], result_sha256=record.get('result_sha256')))
    statuses = [x['status'] for route in ranking['routes'].values() for x in route['all_configurations']]
    stats = {s: statuses.count(s) for s in set(statuses)}
    notes = {
        1: 'Замороженный baseline: прежняя математика, геометрия и фиксированный host cap72.',
        2: 'Адаптивный запуск. Cases2/5/8 остановлены host UB preflight: SIMT UB ошибочно принят за AscendC SIMD UB.',
        3: 'Попытка исправить UB query. До компиляции ядра сработала проверка checksum внешнего build.sh.',
        4: 'Исправлена упаковка. До компиляции ядра выявлено неверное предположение о каталоге metadata SDK.',
        5: 'SDK metadata разрешается относительно реального libplatform.so. Первый успешный адаптивный T.',
        6: 'Повторная загрузка идентичного ZIP отклоненаHTTP400; нового job и списания credit не было. Попытка учтена в лимите12.',
        7: 'Официальный standard reevaluate T; новый submission ID, идентичный SHA256 скачанного архива.',
        8: 'Контрольный standard reevaluate исходного A после двух T.',
        9: 'Не отправлен: финалист F не квалифицирован; остановлено по просьбе пользователя.',
        10: 'Повтор F не отправлен.', 11: 'Финальный контроль A не отправлен.'}
    reviews = []
    for label in ('review-geometry-qwen', 'review-geometry-numerical-qwen', 'review-lowering-qwen'):
        proof = json.loads(read(CAMP / label / 'provenance.json'))
        reviews.append(dict(name=label, **{k: proof.get(k) for k in ('model', 'session_ids', 'completed_review', 'inputs_unchanged_at_completion', 'prompt_file_sha256', 'response_file_sha256', 'skill_delivery')}))
    for rel in ('queue/ledger.json', 'queue/clone-identity-policy.json', 'competitors/ROUND3.md',
                'competitors/manifest.json', 'VF_IR_BOUNDARIES.md'):
        read(CAMP / rel)
    data = dict(captured_at_utc=captured, campaign_pin='adadd7d66ed0ee16d33d79487bf584899a26ef1e',
                stop=dict(user_requested=True, heartbeat_deleted=True, active_remote_jobs=[], active_local_campaign_processes=[],
                          preexisting_rmsnorm_automation='already PAUSED; unchanged'),
                attempts_consumed=8, attempts_cap=12, planned_slots=11, credits=credits,
                sources=sources, navigation=navigation, runs=runs, reviews=reviews, local_ranking=clean(ranking),
                local_phase_records=phases, local_configuration_counts=stats, source_manifest=manifest)
    publish(OUT / 'evidence.json', json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    readme = f'''# GeLU — промежуточный отчёт: адаптация к runner и JIT

Срез: **{captured}**. Кампания10–11сентября2026 остановлена по запросу пользователя.
Таймер удалён; локальных процессов кампании и активных CANNBench jobs на момент
проверки нет. Pending-слоты сохранены, но не будут отправляться автоматически.
Старый отдельный таймер RMSNorm уже имел статусPAUSED и оставлен без изменений.

## Основной результат

**Цель20/20 и GM>1× не достигнута.** Два успешных адаптивных T дали20/20,
GM **0.429502×** и **0.447789×**. В обоих19из20cases медленнее reference.
Контроли A: **0.441481×** и **0.443313×**. Надёжного преимущества только
адаптации запуска эти наблюдения не показали.

В локальном CaModel найдены улучшения FP32 exact и FP16/BF16 exact, но они
**ещё не собраны в квалифицированного финалиста и не измерены на CANNBench**.
FP32tanh JIT-поиск не начат; следующий поиск геометрии и её взаимодействий
с JIT не выполнен. Отчёт промежуточный, а не заявление о завершении плана.

Использовано **8из12 разрешённых попыток**, включая отклонённый duplicate POST;
создано **7аппаратных jobs**. Доступно **{credits['remaining']}credits**;
восстановление по API: `{credits['resets_at']}`. Credits и лимит12попыток — разные
ограничения. Прерванная схемаA–B–B–A превратилась в A–B–R–S–T–reject–T–A
из-за ошибок интеграции; исходный чистый A/B-дизайн не был сохранён.

## Что оценивалось и откуда данные

Оператор **только GeLU**,20официальных cases, FP16/BF16/FP32 × exact/tanh.
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
только при20корректных измеренных cases; failed cases не исключаются ради GM.
Ссылки на jobs могут потребовать входа: это **private submissions**.

Снимки исходников: [A](kernels/A.py), [B](kernels/B.py),
[R/S/T — один kernel source](kernels/T.py), [локальный tuning](kernels/local.py).
Это навигационные снимки модулей, не самостоятельные runnable-пакеты:
host helpers, SDK и evaluator wheels намеренно не включены в публикацию отчёта.
Совпадение kernel SHA с ledger проверено для каждого job.

## Все попытки

'''
    rows = []
    for run in runs:
        s = run.get('summary')
        job = f"[{run['job_id']}]({run['url']})" if run.get('job_id') else '—'
        rows.append([run['slot'], run['candidate'], job, run.get('status', run['state']),
                     f"{s['passed']}/20" if s else 'не измерено', num(s['gm_all20'], 6) if s else '—',
                     run.get('runner', '—')])
    readme += table(['Слот', 'Вариант', 'Job', 'Статус', 'Accuracy', 'GM×', 'Runner'], rows)
    readme += '\n' + '\n'.join(f"- **{i}.** {notes[i]}" for i in range(1, 12)) + '\n'
    readme += '''
## Навигация по20cases и геометрия аппаратных вариантов

Геометрия ниже вычислена из зафиксированных host selectors A/B/T, не угадана
по имени runner. `A blocks/useful/maxTiles` — **реконструкция из исходника**,
не снятая на устройстве telemetry. Для B/T фактически запрошенные vector cores,
resource limits и launched blocks в сохранённом evidence не подтверждены:
**unknown**. Их нельзя заменять числом64/72 по суффиксу SoC.

Для всех remote вариантов requested JIT: reuse_alloc=1,static_alloc=None,
insert_sync=True,opt_level=3,debug=False; VF указан поcase. Effective JIT на
каждом hardware job отдельно не зафиксирован. В локальном проверенном runtime
None соответствует платформенной static allocation; requested/effective
compiler options проверяются в локальном JIT-поиске.

UB ниже — **локальный compiled UB balanced-ядра при той же геометрии/JIT**,
не аппаратный peak/occupancy и не отдельное измерение A. Подтверждённый SDK
профиль CaModel:253952байта UB и72vector cores; это не live query runner.
CaModel timing для каждой полной официальной shape не получен: reduced-N
замеры следующего раздела нельзя подставлять в эти строки.

'''
    readme += table(['Case', 'Shape', 'dtype', 'Mode', 'tile_shape', 'unroll', 'VF', 'A blocks/useful/maxTiles', 'Локальный UB,байт'],
       [[r['case_id'], '×'.join(map(str, r['shape'])), r['dtype'], r['mode'], r['tile_shape'], r['unroll'], r['vf_fusion'],
         f"{r['baseline_requested_blocks']}/{r['baseline_useful_blocks']}/{r['baseline_max_tiles_per_block']}", r['local_balanced_compile_ub_bytes']] for r in navigation])
    for run in runs:
        if not run.get('job_id'): continue
        readme += f"\n## Слот{run['slot']}: {run['candidate']} — [job]({run['url']})\n\n{notes[run['slot']]}\n\n"
        readme += f"[Исходник]({run['kernel']}); SHA256 `{run['source_sha256']}`.\n\n"
        summary = run.get('summary')
        if not summary:
            readme += 'До исполнения20cases не дошло; kernel accuracy, timings и GM не измерены. Это не20численных ошибок.\n'
            continue
        readme += f"Accuracy **{summary['passed']}/20**, anti-cheat failures **{summary['anti_cheat_failed_cases']}**, GM **{num(summary['gm_all20'],6)}×**.\n\n"
        job_rows = []
        for metric in summary['rows']:
            c = navigation[metric['case_id'] - 1]
            job_rows.append([metric['case_id'], '×'.join(map(str,c['shape'])), c['dtype'], c['mode'],
                f"{c['tile_shape']}/u{c['unroll']}", 'pass' if metric['accuracy'] else 'fail',
                num(metric['reference_us']), num(metric['elapsed_us']), num(metric['speedup'])])
        readme += table(['Case','Shape','dtype','Mode','Tile/unroll','Accuracy','Reference,µs','Kernel,µs','Speedup×'], job_rows)
    readme += '''
## Локальная матрица JIT — отдельно от аппаратных результатов

Все строки с двумя timings: N262144,8логических блоков, одинаковая математика
внутри dtype/mode, два свежих процесса; в каждом warmup и3timed repeats.
Единица — **CaModel ticks**, меньше лучше. Compilation/diagnostic overhead
не входит в ticks. Accuracy охватывает reduced stress, численные границы,
длинный хвост, idle-tail, guards и повторное исполнение, не все20полных shapes.
Проверено insert_sync=True,verify_sync=True,opt_level=3,debug=False.
Во всех приведённых представителях static_alloc=None; другие исходные
вариантыFalse/True присутствуют в compile screen, а не проигнорированы.

'''
    local_rows = []
    for route, record in ranking['routes'].items():
        for row in record['all_configurations']:
            c = row['cell']; status = row['status']
            label = 'local accuracy+timing pass' if status == 'qualified_local_ranking' else 'timeout900s; accuracy не установлена' if status == 'rejected' and row.get('failure_class') == 'timeout_no_numerical_conclusion' else 'не запускался'
            local_rows.append([route, f"[{c['tile']}]/u{c['unroll']}", c['reuse'], bool(c['vf']),
                               row.get('ub_bytes','—'), '/'.join(map(str,row.get('timings',[]))) or '—',label])
    readme += table(['Маршрут','Tile/unroll','reuse_alloc','VF','UB,байт','ticks:повтор1/2','Статус'], local_rows)
    readme += f'''
Из25представителей **{stats.get('qualified_local_ranking',0)}** прошли все локальные
гейты, **{stats.get('rejected',0)}** отклонены, **{stats.get('running_or_pending',0)}**
не запускались. Исходный compile screen:108JIT-клеток,50допустимых к дальнейшим
проверкам,20UB-overflow,38unknown/failed. Из50компиляций25эквивалентных
None/True исключены только после совпадения C++,binary и UB. Отдельный
первичный geometry compile screen:96клеток,93допустимы,3overflow; это **не**
новый поиск геометрии с top2JIT, который пока не выполнялся.

### Подтверждённые локальные выводы

- **FP32exact:** reuse0,VFtrue —34915/34915 против41142/41141 уreuse1,VFtrue:
  примерно15,1%меньше ticks при росте UB61440→102400. Отключать VF не нужно.
- **FP16/BF16exact:** reuse2,VFfalse —16853/16852 против≈21837 уreuse1,VFfalse:
  ≈22,8%меньше ticks; UB98304→131072. ДобавлениеVF даёт16631, ещё≈1,3%,
  то есть само по себе не достигает порога5% для продвижения отдельной идеи.
- **FP16/BF16tanh:** reuse2 экономит UB131072→98304, но замедляет9636→10723
  ticks (≈11,3%больше). «Больше свободного UB» не равно «быстрее».
- **FP32exact reuse2,VFtrue:**40157/40156, лишь≈2,4%лучше baseline, хужеreuse0.
  Удаление одного intermediate store само по себе не определяет общий speedup.

## Lowering, конкуренты и ограничения доказательств

В matched FP32exact IR прослежено: FindVFGroup объявляет промежуточное x*0.5
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
tile[32,32], emitted allocation до28800байт; финальный AIV binary/accuracy/perf
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

1. FP32tanh:4локальных JIT-представителя не запущены. Общая матрица не завершена.
2. Top2-per-route поиск tile/unroll/core и повторная JIT-проверка победившей
   геометрии не выполнялись; подготовлены драйверы и31host-test, не hardware evidence.
3. Старый probe `odd-tail=17*tile+1` содержит18тайлов: это long-tail, не odd-count.
   Новый driver задаёт2*blocks+1тайлов, но ещё не запускался. Финалист должен
   пройти настоящее odd-count покрытие, полные specialization и evaluator-package
   гейты с совпадением source/wheel hashes.
4. F не собран/не отправлен; слоты9F,10F,11A остаются неисполненными.
5. Между03:11UTC и запросом остановки в истории есть поступления heartbeat без
   новых подтверждённых действий агента. Последний запущенный bounded batch
   завершил FP32exact, но FP32tanh автоматически не стартовал. Нельзя выдавать
   срабатывание таймера за выполненную работу или утверждать отсутствие простоя.

## Итог

Интеграционные ошибки UB-query/упаковки/SDK-layout исправлены до двух T с20/20.
Переносимость host launch улучшена, но выигрыш только адаптации не доказан.
Перспективны разные JIT-настройки по маршрутам, а не единый reuse_alloc для всех.
Заполнение UB до предела не цель: данные показывают и выигрыш с большим UB,
и замедление при его экономии. Аппаратный GM остаётся≈0,43–0,45×.
Возобновление — только по новому запросу пользователя; credits автоматически
не расходуются, план не объявлен выполненным.
'''
    readme = re.sub(r'(?<=[А-Яа-яЁё])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[А-Яа-яЁё])', ' ', readme)
    for before, after in (('FP32tanh', 'FP32 tanh'), ('FP32exact', 'FP32 exact'),
                          ('20cases', '20 cases'), ('5credits', '5 credits'), ('900s', '900 s')):
        readme = readme.replace(before, after)
    publish(OUT / 'README.md', readme)
    print(json.dumps(dict(report=str(OUT / 'README.md'), stats=stats, phases=len(phases), active_jobs=observed['active_jobs'], credits=credits)))


if __name__ == '__main__':
    if '--export-to' in sys.argv:
        target = Path(sys.argv[sys.argv.index('--export-to') + 1]).resolve()
        assert (target / '.git').exists(), 'Destination must be a Git worktree'
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix in ('.md', '.json', '.py') and '__pycache__' not in path.parts:
                publish(target / 'reports' / OUT.name / path.relative_to(OUT), path.read_text())
        print('Exported report-only files into ' + str(target))
    else:
        main()
