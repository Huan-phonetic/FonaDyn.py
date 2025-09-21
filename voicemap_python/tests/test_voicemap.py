#!/usr/bin/env python3
"""
VoiceMap Test Suite
Basic tests for the VoiceMap analyzer
"""

import sys
import os
import unittest
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyzer import VoiceMapAnalyzer
from config import VoiceMapConfig
from metrics import SPLCalculator, ClarityCalculator


class TestVoiceMapConfig(unittest.TestCase):
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = VoiceMapConfig()
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.clarity_threshold, 0.96)
        self.assertEqual(config.spl_correction_db, 120.0)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = VoiceMapConfig(
            sample_rate=48000,
            clarity_threshold=0.95,
            spl_correction_db=100.0
        )
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.clarity_threshold, 0.95)
        self.assertEqual(config.spl_correction_db, 100.0)


class TestMetricCalculators(unittest.TestCase):
    """Test metric calculators"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = VoiceMapConfig()
        self.spl_calculator = SPLCalculator(self.config)
        self.clarity_calculator = ClarityCalculator(self.config)
    
    def test_spl_calculator_initialization(self):
        """Test SPL calculator initialization"""
        self.assertIsNotNone(self.spl_calculator)
        self.assertEqual(self.spl_calculator.sample_rate, 44100)
    
    def test_clarity_calculator_initialization(self):
        """Test Clarity calculator initialization"""
        self.assertIsNotNone(self.clarity_calculator)
        self.assertEqual(self.clarity_calculator.sample_rate, 44100)


class TestVoiceMapAnalyzer(unittest.TestCase):
    """Test main analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = VoiceMapAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.config)
        self.assertIsNotNone(self.analyzer.spl_calculator)
        self.assertIsNotNone(self.analyzer.clarity_calculator)
    
    def test_peak_follower(self):
        """Test peak follower algorithm"""
        signal = np.array([1, 2, 1, 3, 2, 4, 3, 2, 1])
        decay = 0.9
        result = self.analyzer.peak_follower(signal, decay)
        
        self.assertEqual(len(result), len(signal))
        self.assertTrue(np.all(result >= 0))
    
    def test_fos_filter(self):
        """Test FOS filter"""
        signal = np.array([1, 2, 3, 4, 5])
        coeff = 0.5
        result = self.analyzer.fos_filter(signal, coeff)
        
        self.assertEqual(len(result), len(signal))
    
    def test_set_reset_ff(self):
        """Test set-reset flip-flop"""
        set_signal = np.array([1, 0, 1, 0, 1])
        reset_signal = np.array([0, 1, 0, 1, 0])
        result = self.analyzer.set_reset_ff(set_signal, reset_signal)
        
        self.assertEqual(len(result), len(set_signal))
        self.assertTrue(np.all((result == 0) | (result == 1)))


if __name__ == '__main__':
    unittest.main()
