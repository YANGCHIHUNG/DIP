import numpy as np
from src.matcher import match_features

def test_match_features():
    # 產生隨機描述子（10 個 128 維向量）
    desc = np.random.rand(10, 128).astype('float32')
    # 與自身匹配
    matches = match_features(desc, desc)
    
    # 檢查回傳型態
    assert isinstance(matches, list)
    # 每個 match 應該具有 queryIdx 和 trainIdx 屬性
    assert all(hasattr(m, 'queryIdx') and hasattr(m, 'trainIdx') for m in matches)
