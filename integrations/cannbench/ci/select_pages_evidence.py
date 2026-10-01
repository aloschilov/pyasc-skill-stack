#!/usr/bin/env python3
"""Select an exact measured artifact without relying on all-workflow pagination."""
import json
import os
import urllib.error
import urllib.request


def select(runs, source_sha):
    return next((run for run in runs if run['head_sha'] == source_sha), runs[0] if runs else None)


def main():
    if os.environ.get('EVENT_WORKFLOW') == 'CANNBench gate':
        run_id, source = os.environ['EVENT_RUN_ID'], os.environ['EVENT_SHA']
    else:
        repository = os.environ['GITHUB_REPOSITORY']
        url = f'https://api.github.com/repos/{repository}/actions/workflows/cannbench.yml/runs?branch=main&status=completed&per_page=100'
        request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'], 'Accept': 'application/vnd.github+json'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                runs = json.load(response)['workflow_runs']
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            runs = []  # First deployment before the workflow is registered.
        selected = select(runs, os.environ['SOURCE_SHA'])
        run_id = str(selected['id']) if selected else ''
        source = selected['head_sha'] if selected else ''
        print(f'Inspected {len(runs)} completed CANNBench runs')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'run_id={run_id}\nsource_sha={source}\n')
    print(f'Selected hardware evidence run {run_id or "none"}, revision {source or "none"}')


if __name__ == '__main__':
    main()
