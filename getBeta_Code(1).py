import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def run_mcm_prior_pipeline(file_path, rf_threshold=0.005):
    # --- 1. 数据预处理 ---
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['placement'])
    
    # 计算评委平均分 (舞蹈硬实力代理指标)
    score_cols = [c for c in df.columns if 'judge' in c]
    judge_avg = df[score_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0)
    
    # 计算粉丝倾向代理指标 (Residual: 实际表现 - 评分表现)
    # 归一化到 [0, 1]，值越大代表粉丝加持越高
    actual_score = (df['placement'].max() - df['placement']) / (df['placement'].max() - df['placement'].min() + 1e-8)
    judge_score_norm = (judge_avg - judge_avg.min()) / (judge_avg.max() - judge_avg.min() + 1e-8)
    fan_proxy = actual_score - judge_score_norm

    # --- 2. 特征工程 ---
    scan_map = {
        'celebrity_industry': 'Occupation',
        'ballroom_partner': 'Partner',
        'celebrity_homestate': 'State'
    }
    
    # 处理年龄
    df['Age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce').fillna(df['celebrity_age_during_season'].mean())
    
    # 独热编码
    features_raw = pd.get_dummies(df[list(scan_map.keys())].fillna('Unknown'))
    features_raw['Age'] = df['Age']
    
    # --- 3. 随机森林筛选 (Feature Selection) ---
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf.fit(features_raw, fan_proxy)
    
    importances = pd.Series(rf.feature_importances_, index=features_raw.columns)
    # 筛选掉贡献度太低的特征，避免 GLM 过拟合
    selected_features = importances[importances > rf_threshold].index.tolist()
    
    print(f"RF 筛选完成: 从 {len(features_raw.columns)} 个特征中保留了 {len(selected_features)} 个。")

    # --- 4. GLM 建立先验 (Beta Extraction) ---
    X_selected = features_raw[selected_features]
    
    # 标准化：保证 Age 和 Binary 特征的 Beta 在同一量级，可直接比较
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    glm = Ridge(alpha=1.0) 
    glm.fit(X_scaled, fan_proxy)
    
    # 整理结果
    beta_results = pd.DataFrame({
        'Full_Feature': selected_features,
        'Beta': glm.coef_
    })

    # --- 5. 分类输出 CSV ---
    # 输出 RF 筛选后的 Beta
    for original_col, label in scan_map.items():
        mask = beta_results['Full_Feature'].str.startswith(original_col)
        sub_df = beta_results[mask].copy()
        if not sub_df.empty:
            sub_df['Feature'] = sub_df['Full_Feature'].str.replace(f'{original_col}_', '')
            sub_df = sub_df[['Feature', 'Beta']].sort_values('Beta', ascending=False)
            sub_df.to_csv(f'Prior_Beta_{label}.csv', index=False)
            print(f"已生成先验系数: Prior_Beta_{label}.csv")

    # 年龄单独处理
    if 'Age' in selected_features:
        age_beta = beta_results[beta_results['Full_Feature'] == 'Age']
        age_beta[['Full_Feature', 'Beta']].to_csv('Prior_Beta_Demographic.csv', index=False)
        print("已生成先验系数: Prior_Beta_Demographic.csv")

    return beta_results

# 运行流水线
res = run_mcm_prior_pipeline('2026_MCM_Problem_C_Data.csv')