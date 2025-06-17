import unittest
import numpy as np
import librosa
import os
from src.core.vrp_creator import create_VRP_from_Voice_EGG
from src.preprocessing.voice_preprocessor import preprocess_voice_signal
from src.preprocessing.egg_preprocessor import preprocess_egg_signal
from src.metrics.voice_metrics import find_f0, find_spl
from src.metrics.egg_metrics import find_qci, find_deggmax

class TestBasicFunctionality(unittest.TestCase):
    def setUp(self):
        # 加载测试音频文件
        self.audio_file = 'audio/test_Voice_EGG.wav'
        self.signal, self.sr = librosa.load(self.audio_file, sr=44100, mono=False)

    def test_voice_preprocessing(self):
        """测试语音预处理功能"""
        processed_voice = preprocess_voice_signal(self.signal[0], self.sr)
        self.assertIsNotNone(processed_voice)
        self.assertEqual(len(processed_voice), len(self.signal[0]))

    def test_egg_preprocessing(self):
        """测试EGG预处理功能"""
        processed_egg = preprocess_egg_signal(self.signal[1], self.sr)
        self.assertIsNotNone(processed_egg)
        self.assertEqual(len(processed_egg), len(self.signal[1]))

    def test_voice_metrics(self):
        """测试语音指标计算"""
        f0, _ = find_f0(self.signal[0], self.sr, len(self.signal[0]), len(self.signal[0])//2)
        spl = find_spl(self.signal[0])
        self.assertIsNotNone(f0)
        self.assertIsNotNone(spl)

    def test_egg_metrics(self):
        """测试EGG指标计算"""
        qci = find_qci(self.signal[1])
        deggmax = find_deggmax(self.signal[1])
        self.assertIsNotNone(qci)
        self.assertIsNotNone(deggmax)

    def test_vrp_creation(self):
        """测试VRP创建功能"""
        vrp = create_VRP_from_Voice_EGG(self.signal, self.sr)
        print("\nVRP输出示例:")
        for k, v in vrp.items():
            print(f"{k}: {v[:5]} ... 共{len(v)}项")
        self.assertIsNotNone(vrp)
        self.assertIsInstance(vrp, dict)
        self.assertIn('frequencies', vrp)
        self.assertIn('SPLs', vrp)

if __name__ == '__main__':
    unittest.main() 