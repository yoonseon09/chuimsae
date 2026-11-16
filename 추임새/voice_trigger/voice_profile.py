"""Wav2Vec2 음성 특성 추정. 음성은 로컬 CPU에서만 처리한다.

모델의 male/female은 음원의 라우팅 라벨이다. 신원/성별 확정이나
겹친 발화의 분리 기능이 아니며, 한국어 추임새의 정확도는 현장 검증이 필요하다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from trigger_core import SR

VOICE_MODEL = 'prithivMLmods/Common-Voice-Gender-Detection'
VOICE_REVISION = 'a0f01051e13f3fe6f7afc411ab88cf83bc3fc53a'
MODEL_FILES = ['config.json', 'preprocessor_config.json', 'model.safetensors']


def model_folder():
    from huggingface_hub import hf_hub_download
    kwargs = dict(repo_id=VOICE_MODEL, revision=VOICE_REVISION)
    folder = None
    for filename in MODEL_FILES:
        try:
            path = hf_hub_download(**kwargs, filename=filename, local_files_only=True)
        except OSError:
            try:
                # 필요한 파일만 받는다. 이미 받은 큰 가중치는 다시 조회하지 않는다.
                path = hf_hub_download(**kwargs, filename=filename)
            except Exception as exc:
                detail = str(exc).splitlines()[0][:400] if str(exc) else type(exc).__name__
                raise RuntimeError(f'목소리 모델 파일 {filename} 다운로드 실패 '
                                   f'({type(exc).__name__}: {detail}). '
                                   '이미 받은 파일은 유지됩니다. 연결을 확인한 뒤 다시 실행하세요.') from exc
        current = Path(path).parent
        if folder is not None and current != folder:
            raise RuntimeError('목소리 모델 파일의 리비전 경로가 서로 다릅니다')
        folder = current
    return str(folder)


def prediction(scores, id2label, minimum):
    """분류 점수는 실제 정답 확률로 보증하지 않는다."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (2,) or not np.isfinite(scores).all() or np.any(scores < 0) or not np.isclose(scores.sum(), 1):
        return dict(label='unknown', score=None, reason='invalid_model_output')
    labels = [id2label[i] for i in range(2)]
    if set(labels) != {'male', 'female'}:
        raise ValueError('목소리 모델의 male/female 라벨 매핑이 올바르지 않습니다')
    index = int(np.argmax(scores))
    score = float(scores[index])
    return dict(label=labels[index] if score >= minimum else 'unknown',
                score=score, reason='classified' if score >= minimum else 'low_voice_score')


class VoiceClassifier:
    def __init__(self, settings):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        self.settings = settings
        self.torch = torch
        torch.set_num_threads(2)
        folder = model_folder()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            folder, local_files_only=True, use_safetensors=True).to('cpu').eval()
        self.labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if set(self.labels) != {0, 1} or set(self.labels.values()) != {'female', 'male'}:
            raise ValueError('목소리 모델에는 female/male 두 라벨이 필요합니다')
        if self.processor.sampling_rate != SR:
            raise ValueError('목소리 모델은 16 kHz 입력을 지원해야 합니다')
        # 최초 추론 준비를 마이크 시작 전에 끝낸다.
        self._predict(np.zeros(SR, np.float32))

    def _predict(self, audio):
        inputs = self.processor(audio, sampling_rate=SR, return_tensors='pt', padding=True)
        with self.torch.inference_mode():
            scores = self.model(**inputs).logits.softmax(dim=-1)[0].cpu().numpy()
        return prediction(scores, self.labels, self.settings['voice_min_confidence'])

    def classify(self, audio):
        result = dict(label='unknown', method='wav2vec2', model=VOICE_MODEL,
                      score=None, active_seconds=0.0, reason='insufficient_audio')
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            return result
        # 발화 전후의 무음만 자른다. 내부의 자음/쉼은 이어 붙이지 않는다.
        hop = int(SR * 0.01)
        count = len(audio) // hop
        if not count:
            return result
        frames = audio[:count * hop].reshape(count, hop)
        energy = np.sqrt(np.mean((frames - frames.mean(axis=1, keepdims=True)) ** 2, axis=1))
        threshold = max(10 ** (-45 / 20), float(energy.max()) * 0.1)
        active = np.flatnonzero(energy >= threshold)
        result['active_seconds'] = round(len(active) * hop / SR, 4)
        if not len(active) or result['active_seconds'] < self.settings['voice_min_seconds']:
            return result
        start = max(0, (active[0] - 2) * hop)
        end = min(len(audio), (active[-1] + 3) * hop)
        if end - start > 30 * SR:
            raise ValueError('목소리 검사는 한 사람이 말하는 30초 이하 녹음을 사용하세요')
        result.update(self._predict(audio[start:end]))
        return result
