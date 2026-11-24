"""음원 혼선·낮은 점수·모델 실패를 검사한다. 실제 목소리 정확도 검사가 아니다."""
import json
import tempfile
import unittest
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from trigger_core import Command, Config, DEFAULTS, SR, Utterance, route_command
from voice_profile import VoiceClassifier, model_folder, prediction
from voice_trigger import Player, recognition_worker, record_result


class VoiceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(DEFAULTS, voice_mode='model')
        self.command = Command('얼쑤', ('얼쑤',), (Path('legacy.wav'),),
                               voice_sounds={'male': (Path('남_얼쑤.wav'),),
                                             'female': (Path('여_얼쑤.wav'),)})

    def test_route_never_uses_other_group_or_legacy(self):
        for label in ('male', 'female'):
            routed, reason = route_command(self.command, {'label': label}, self.settings)
            self.assertEqual(reason, 'accepted')
            self.assertEqual(routed.sounds, self.command.voice_sounds[label])
        for voice in (None, {}, {'label': 'unknown'}, {'label': 'other'}):
            self.assertEqual(route_command(self.command, voice, self.settings), (None, 'voice_unknown'))
        male_only = Command('얼쑤', ('얼쑤',), (Path('legacy.wav'),),
                            voice_sounds={'male': (Path('남_얼쑤.wav'),)})
        self.assertEqual(route_command(male_only, {'label': 'female'}, self.settings),
                         (None, 'voice_sound_missing'))
        self.assertEqual(route_command(self.command, None, DEFAULTS), (self.command, 'accepted'))

    def test_model_mapping_and_score_gate(self):
        self.assertEqual(prediction([.9,.1], {0:'female',1:'male'}, .8)['label'], 'female')
        self.assertEqual(prediction([.9,.1], {0:'male',1:'female'}, .8)['label'], 'male')
        self.assertEqual(prediction([.7,.3], {0:'female',1:'male'}, .8)['label'], 'unknown')
        for values in ([np.nan,.2], [.8], [-.1,1.1], [.8,.8]):
            self.assertEqual(prediction(values, {0:'female',1:'male'}, .8)['label'], 'unknown')
        with self.assertRaises(ValueError):
            prediction([.9,.1], {0:'LABEL_0',1:'LABEL_1'}, .8)

    def test_config_discovers_unicode_variants_and_missing_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('남_얼쑤.wav','남_얼쑤2.wav','여_좋다.wav','여_얼쑤.wav.bak'):
                (root / unicodedata.normalize('NFD',name)).touch()
            data={'settings': self.settings, 'commands': [dict(name='얼쑤', phrases=['얼쑤'],
                  sounds=['남_얼쑤.wav'], voice_sounds={'male':['남_얼쑤*.wav'], 'female':['여_얼쑤*.wav']})]}
            path=root/'commands.json'
            path.write_text(json.dumps(data))
            command=Config.load(path).commands[0]
            self.assertEqual(len(command.voice_sounds['male']),2)
            self.assertEqual(command.voice_sounds['female'],())
            self.assertEqual(len(command.all_sounds),2)
            (root / '여_얼쑤.wav').touch()
            self.assertEqual(len(Config.load(path).commands[0].voice_sounds['female']),1)
            for invalid in ({'voice_min_confidence':.5}, {'voice_min_seconds':True},
                            {'voice_mode':'pitch'}, {'voice_min_confidence':float('nan')}):
                data['settings']=invalid
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    Config.load(path)
            data['settings']=self.settings
            data['commands'][0]['voice_sounds']={'female':['../other.wav']}
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                Config.load(path)

    def test_result_routing_and_log_metadata(self):
        cfg=SimpleNamespace(settings=self.settings,commands=(self.command,))
        events=SimpleNamespace(emit=MagicMock())
        item=Utterance(np.zeros(SR,np.float32),0,1,.5,uid=3)
        result={'text':'얼쑤', 'voice':{'label':'female','score':.95}, 'segments':[
            dict(text='얼쑤',avg_logprob=-.1,no_speech_prob=.01,compression_ratio=1)]}
        with patch('voice_trigger.log'):
            command=record_result(item,('result',3,result,.1),cfg,events)
        self.assertEqual(command.sounds,(Path('여_얼쑤.wav'),))
        self.assertEqual(events.emit.call_args.kwargs['voice']['label'],'female')
        self.assertEqual(events.emit.call_args.kwargs['sound_candidates'],['여_얼쑤.wav'])
        result['voice']['label']='unknown'
        with patch('voice_trigger.log'):
            self.assertIsNone(record_result(item,('result',3,result,.1),cfg,events))
        self.assertEqual(events.emit.call_args.kwargs['reason'],'voice_unknown')

    def test_preload_and_schedule_select_the_routed_wav(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            t=np.arange(3200)/SR
            for name,frequency in [('male.wav',130),('female.wav',240)]:
                sf.write(root/name,.2*np.sin(2*np.pi*frequency*t),SR)
            command=Command('얼쑤',('얼쑤',),(root/'male.wav',),voice_sounds={
                'male':(root/'male.wav',),'female':(root/'female.wav',)})
            cfg=SimpleNamespace(settings=self.settings,commands=(command,))
            with patch('voice_trigger.log'):
                player=Player(cfg,dry=True)
            self.assertEqual(len(player.cache),2)
            routed,_=route_command(command,{'label':'female'},self.settings)
            with patch('voice_trigger.time.monotonic',return_value=10):
                until=player.schedule(Utterance(np.empty(0),9,9.9,.2,uid=1),routed)
            self.assertIsNotNone(until)
            event,fields=player.notifications.get_nowait()
            self.assertEqual((event,fields['sound']),('sound_selected','female.wav'))


class VoiceModelTests(unittest.TestCase):
    def classifier_without_weights(self):
        classifier=VoiceClassifier.__new__(VoiceClassifier)
        classifier.settings=dict(DEFAULTS)
        classifier._predict=MagicMock(return_value=dict(label='female',score=.95,reason='classified'))
        return classifier

    def test_empty_quiet_short_and_nonfinite_skip_model(self):
        classifier=self.classifier_without_weights()
        for audio in (np.zeros(SR),np.ones(SR),np.empty(0),np.full(SR,np.nan),
                      .1*np.sin(2*np.pi*200*np.arange(1000)/SR)):
            self.assertEqual(classifier.classify(audio)['label'],'unknown')
        classifier._predict.assert_not_called()

    def test_trims_outer_silence_preserves_internal_pause(self):
        classifier=self.classifier_without_weights()
        tone=.1*np.sin(2*np.pi*200*np.arange(3200)/SR)
        clip=np.concatenate([np.zeros(4000),tone,np.zeros(1000),tone,np.zeros(4000)])
        self.assertEqual(classifier.classify(clip)['label'],'female')
        processed=classifier._predict.call_args.args[0]
        self.assertGreater(len(processed),len(tone)*2+1000)
        self.assertLess(len(processed),len(clip))

    def test_download_failure_does_not_silently_use_pitch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('huggingface_hub.hf_hub_download',side_effect=[OSError('missing'),OSError('offline')]) as download:
                with self.assertRaisesRegex(RuntimeError,'config.json.*offline'):
                    model_folder()
            self.assertEqual(download.call_count,2)
            self.assertTrue(download.call_args_list[0].kwargs['local_files_only'])

    def test_cached_complete_model_needs_no_network(self):
        from voice_profile import MODEL_FILES
        with tempfile.TemporaryDirectory() as tmp:
            for name in MODEL_FILES:
                Path(tmp,name).touch()
            with patch('huggingface_hub.hf_hub_download',side_effect=[str(Path(tmp,name)) for name in MODEL_FILES]) as download:
                self.assertEqual(model_folder(),tmp)
            self.assertEqual(download.call_count,3)
            self.assertTrue(all(call.kwargs['local_files_only'] for call in download.call_args_list))

    def test_missing_config_download_preserves_cached_weights(self):
        from voice_profile import MODEL_FILES
        with tempfile.TemporaryDirectory() as tmp:
            paths=[str(Path(tmp,name)) for name in MODEL_FILES]
            with patch('huggingface_hub.hf_hub_download',side_effect=[OSError('missing'),*paths]) as download:
                self.assertEqual(model_folder(),tmp)
            calls=download.call_args_list
            self.assertEqual([c.kwargs['filename'] for c in calls],
                             ['config.json','config.json','preprocessor_config.json','model.safetensors'])
            self.assertNotIn('local_files_only',calls[1].kwargs)
            self.assertTrue(calls[-1].kwargs['local_files_only'])

    def test_worker_propagates_voice_result(self):
        settings=dict(DEFAULTS,voice_mode='model')
        connection=MagicMock()
        connection.recv.side_effect=[Utterance(np.ones(SR,np.float32),0,1,1,uid=7),None]
        engine=SimpleNamespace(transcribe=MagicMock(return_value={'text':'얼쑤','segments':[]}))
        classifier=MagicMock()
        classifier.classify.return_value=dict(label='male',score=.9)
        with patch.dict('sys.modules',{'mlx_whisper':engine}), \
             patch('voice_trigger.local_model',return_value='local'), \
             patch('voice_trigger.VoiceClassifier',return_value=classifier):
            recognition_worker(connection,settings)
        self.assertEqual(connection.send.call_args_list[1].args[0][2]['voice']['label'],'male')

    def test_real_transformers_load_and_forward_with_tiny_local_model(self):
        # 임의 가중치로 API/전처리/추론 호환성만 확인. 학습 모델의 정확도 검사가 아니다.
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        config=Wav2Vec2Config(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
            intermediate_size=32,conv_dim=(8,8,8),conv_kernel=(10,3,3),conv_stride=(5,2,2),
            num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=16,
            id2label={0:'female',1:'male'},label2id={'female':0,'male':1})
        with tempfile.TemporaryDirectory() as tmp:
            Wav2Vec2ForSequenceClassification(config).save_pretrained(tmp,safe_serialization=True)
            Wav2Vec2FeatureExtractor(sampling_rate=SR).save_pretrained(tmp)
            with patch('voice_profile.model_folder',return_value=tmp):
                classifier=VoiceClassifier(DEFAULTS)
            result=classifier.classify(.1*np.sin(2*np.pi*200*np.arange(6400)/SR))
            self.assertEqual(result['method'],'wav2vec2')
            self.assertIsNotNone(result['score'])
            self.assertIn(result['label'],('male','female','unknown'))


if __name__=='__main__':
    unittest.main()
