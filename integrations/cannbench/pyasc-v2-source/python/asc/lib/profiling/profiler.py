# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

from .. import runtime as rt
from ..utils import get_ascend_path
from .msprof import AicoreMetrics, MsprofInterface, ProfilerConfig, ProfileType
from .result import ProfilingResult, ProfilingTask


class Profiler:

    def __init__(self, result_path: Optional[str] = None):
        self.msprof = MsprofInterface()
        self.config: Optional[ProfilerConfig] = None
        self.started = False
        self.result_path = result_path
        self.auto_remove = result_path is None
        self.results: Dict[str, ProfilingResult] = {}

    @property
    def last_result(self) -> ProfilingResult:
        if not self.results:
            raise RuntimeError("No results were stored, maybe profilng was never finished")
        last_id = next(reversed(self.results.keys()))
        return self.results[last_id]

    @classmethod
    def populate_tasks(cls, filename: str, col_id: str, col_name: str, col_type: str, col_duration: str,
                       tasks: List[ProfilingTask]) -> None:
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = ProfilingTask(
                    id=int(row[col_id]),
                    name=row[col_name],
                    type=row[col_type],
                    duration=float(row[col_duration]),
                )
                tasks.append(task)

    def start(self, device_id: Optional[int] = None) -> None:
        if device_id is None:
            device_id = rt.current_device()
        if self.result_path is None:
            self.result_path = tempfile.mkdtemp(prefix="pyasc_profiler_")
        else:
            self.result_path = str(Path(self.result_path).resolve())
        self.msprof.init(self.result_path)
        self.config = self.msprof.create_config(
            [device_id], AicoreMetrics.PIPE_UTILIZATION,
            [ProfileType.TASK_TIME, ProfileType.AICORE_METRICS, ProfileType.L2CACHE])
        self.msprof.start(self.config)
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        self.msprof.stop(self.config)
        self.msprof.destroy_config(self.config)
        self.config = None
        self.msprof.finalize()

    def export(self) -> None:
        msprof_tool = get_ascend_path() / "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        if not msprof_tool.is_file():
            raise RuntimeError(f"msprof tool does not exist at {msprof_tool}")
        for arg in ("summary", "timeline"):
            cmd = [sys.executable, str(msprof_tool), "export", arg, "-dir", self.result_path]
            subprocess.run(cmd, capture_output=True, check=True)

    def cleanup(self) -> None:
        if self.auto_remove and os.path.isdir(self.result_path):
            shutil.rmtree(self.result_path)
            self.result_path = None

    def store_result(self, run_id: str) -> ProfilingResult:
        result_dir = Path(self.result_path) / run_id
        if not result_dir.is_dir():
            raise RuntimeError(f"{result_dir} is not a directory")
        csv_files = tuple(result_dir.glob("**/*.csv"))
        if not csv_files:
            raise RuntimeError(f"There is no CSV reports in {result_dir}")
        tasks = []
        for item in csv_files:
            if item.name.startswith("op_summary_"):
                self.populate_tasks(item, "Task ID", "Op Name", "Task Type", "Task Duration(us)", tasks)
                break
            if item.name.startswith("task_time_"):
                self.populate_tasks(item, "task_id", "kernel_name", "kernel_type", "task_time(us)", tasks)
                break
        self.results[run_id] = ProfilingResult(tasks, run_id, stored_at=datetime.now())

    def store_last_result(self) -> ProfilingResult:
        if not os.path.isdir(self.result_path):
            raise RuntimeError(f"{self.result_path} is not a directory")
        last_run_id = max((d for d in os.listdir(self.result_path)), default=None)
        if last_run_id is None:
            raise RuntimeError(f"There is no profiling results in {self.result_path}")
        self.store_result(last_run_id)

    @contextmanager
    def profile(self, device_id: Optional[int] = None, cleanup: bool = True):
        try:
            self.start(device_id)
            yield
            self.stop()
            self.export()
            self.store_last_result()
        finally:
            if cleanup:
                self.cleanup()
