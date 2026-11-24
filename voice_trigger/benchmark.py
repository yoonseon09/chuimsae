#!/usr/bin/env python3
"""정답이 있는 파일 묶음의 후보 감지율 검사. 실제 출력 지연을 측정하지 않는다."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf

from trigger_core import Config, resolve_path
from voice_trigger import HERE, Recognizer, Silero, analyze_file, log


class Collector:
    def __init__(self):
        self.rows = []

    def emit(self, event, **fields):
        self.rows.append({'event': event, **fields})


def load_manifest(path, cfg):
    entries = []
    registered = {command.name for command in cfg.commands}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict) or not isinstance(entry.get('audio'), str):
            raise ValueError(f'manifest {number}행: audio 경로가 필요합니다')
        if 'expected' not in entry:
            raise ValueError(f'manifest {number}행: expected를 명시하세요 (대상 없으면 null)')
        expected = entry['expected']
        if expected is not None and expected not in registered:
            raise ValueError(f'manifest {number}행: 활성 명령 이름 또는 null이 필요합니다')
        audio = resolve_path(path.parent, entry['audio'])
        info = sf.info(audio)
        if info.duration <= 0:
            raise ValueError(f'비어 있는 녹음: {audio}')
        entries.append((audio, expected, info.duration))
    if not entries:
        raise ValueError('manifest에 녹음 항목이 없습니다')
    return entries


def summarize(rows):
    positives = [row for row in rows if row['expected'] is not None]
    negatives = [row for row in rows if row['expected'] is None]
    correct = sum(row['expected'] in row['predicted'] for row in positives)
    negative_events = sum(len(row['predicted']) for row in negatives)
    accepted = sum(len(row['predicted']) for row in rows)
    inference = [value for row in rows for value in row['inference_seconds']]
    negative_duration = sum(row['duration_seconds'] for row in negatives)
    return {
        'mode': 'offline_candidate_detection_not_realtime_output',
        'clips': len(rows), 'positive_clips': len(positives), 'negative_clips': len(negatives),
        'true_positive_clips': correct, 'false_negative_clips': len(positives) - correct,
        'false_positive_negative_clips': sum(bool(row['predicted']) for row in negatives),
        'extra_or_wrong_candidate_events': accepted - correct - negative_events,
        'recall': correct / len(positives) if positives else None,
        'candidate_precision': correct / accepted if accepted else None,
        'false_candidate_events_per_negative_hour': negative_events * 3600 / negative_duration if negative_duration else None,
        'median_inference_seconds': statistics.median(inference) if inference else None,
        'max_inference_seconds': max(inference) if inference else None,
        'actual_output_latency_measured': False,
        'results': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--config', type=Path, default=HERE / 'commands.json')
    parser.add_argument('--report', type=Path, default=HERE / 'logs/benchmark.json')
    args = parser.parse_args()
    cfg = Config.load(args.config)
    entries = load_manifest(args.manifest.resolve(), cfg)
    vad = Silero()
    recognizer = Recognizer(cfg.settings)
    rows = []
    try:
        for audio, expected, duration in entries:
            vad.reset()
            collector = Collector()
            analyze_file(audio, cfg, collector, recognizer, vad)
            if any(row['event'] == 'recognition_error' for row in collector.rows):
                raise RuntimeError(f'인식 오류로 비교를 완료할 수 없습니다: {audio}')
            recognized = [row for row in collector.rows if row['event'] == 'recognized']
            predicted = [name for row in recognized if row['reason'] == 'accepted'
                         for name in row['commands']]
            rows.append(dict(audio=str(audio), expected=expected, predicted=predicted,
                             duration_seconds=duration,
                             inference_seconds=[row['inference_seconds'] for row in recognized]))
    finally:
        recognizer.close()
    report = summarize(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    log(f'보고서 저장: {args.report.resolve()}')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log('검사를 종료했습니다.')
    except (OSError, ValueError, RuntimeError, EOFError, ImportError) as exc:
        log(f'검사 실패: {exc}')
        raise SystemExit(1)
