import pandas as pd
import numpy as np
from getBeta_Code import res

def run_bayesian_update_final(file_path, prior_results):
    df = pd.read_csv(file_path)
    
    # 1. 定义 DWTS 规则判定 [cite: 19, 86, 97]
    def get_voting_rule(season):
        # 第1, 2季及第28季以后使用排名制 [cite: 86, 108]
        if season <= 2 or season >= 28:
            return 'Rank'
        # 第3季至第27季使用百分比制 [cite: 97]
        return 'Percentage'

    # 2. 预处理：生存周数与评委表现
    score_cols = [c for c in df.columns if 'judge' in c]
    df['judge_avg'] = df[score_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0)
    
    def parse_survival(res_str):
        if pd.isna(res_str) or 'Winner' in res_str or 'Runner-up' in res_str: return 12
        import re
        nums = re.findall(r'\d+', res_str)
        return int(nums[0]) if nums else 1
    df['survival_weeks'] = df['results'].apply(parse_survival)

    # 3. 提取证据信号 (Evidence Signal)
    # 考虑不同规则对“粉丝力”的转化差异 [cite: 13, 21]
    df['evidence_signal'] = 0.0
    for season in df['season'].unique():
        s_mask = df['season'] == season
        season_df = df[s_mask].copy()
        rule = get_voting_rule(season)
        
        j_rank = season_df['judge_avg'].rank(pct=True) # 评委分越高，排名越靠前
        s_rank = season_df['survival_weeks'].rank(pct=True) # 活得越久，排名越靠前
        
        if rule == 'Rank':
            # 排名制下，粉丝的作用是弥补技术分排名 [cite: 93]
            df.loc[s_mask, 'evidence_signal'] = s_rank - j_rank
        else:
            # 百分比制下，极端的“德不配位”现象（如 Bobby Bones）信号更强 [cite: 15, 102]
            df.loc[s_mask, 'evidence_signal'] = (s_rank - j_rank) * 1.2

    # 4. 后验 Beta 计算
    prior_betas = prior_results.set_index('Full_Feature')['Beta'].to_dict()
    scan_map = {'celebrity_industry': 'Occupation', 'ballroom_partner': 'Partner', 'celebrity_homestate': 'State'}
    
    # 特征构建 (需与 getBeta_Code 保持一致)
    cat_features = pd.get_dummies(df[list(scan_map.keys())].fillna('Unknown'))
    df['Age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce').fillna(df['celebrity_age_during_season'].mean())
    X = pd.concat([cat_features, df[['Age']]], axis=1)

    learning_rate = 0.1
    posterior_list = []

    # --- 建议修改的代码段 ---
    for feature_col in X.columns:
        if feature_col in prior_betas:
            mask = (X[feature_col] > 0)
            sample_size = mask.sum()
            
            if sample_size > 0:
                avg_evidence = df.loc[mask, 'evidence_signal'].mean()
                
                # S28+ 修正逻辑保持不变
                recent_ratio = df.loc[mask, 'season'].apply(lambda x: 1 if x >= 28 else 0).mean()
                adjusted_lr = learning_rate * (1 - 0.4 * recent_ratio) 
                
                # 【核心修改】：引入 Logistic/Tanh 平滑抑制过拟合
                # 使用 np.tanh 可以将任意范围的证据压缩在 [-1, 1] 之间
                # 这样即使某个特征的 evidence_signal 是 5.0，也会被平滑处理
                smoothed_signal = np.tanh(avg_evidence) 
                
                # 【核心修改】：引入基于样本量的可信度惩罚 (shrinkage)
                # 只有当样本量增加时，smoothed_signal 才能发挥 100% 的作用
                sample_weight = sample_size / (sample_size + 5) # 这里的 5 是平滑常数，可调
                
                new_beta = prior_betas[feature_col] + (adjusted_lr * smoothed_signal * sample_weight)
                
                # 确定性计算（保持或微调）
                certainty = 1 - (1 / (1 + sample_size))
                
                posterior_list.append({
                    'Full_Feature': feature_col,
                    'Prior_Beta': prior_betas[feature_col],
                    'Posterior_Beta': new_beta,
                    'Certainty': certainty
                })

    posterior_df = pd.DataFrame(posterior_list)

    # 5. 按照 graft.py 要求的格式导出 CSV
    for original_col, label in scan_map.items():
        sub = posterior_df[posterior_df['Full_Feature'].str.startswith(original_col)].copy()
        sub['Feature'] = sub['Full_Feature'].str.replace(f'{original_col}_', '')
        # 导出绘图函数所需的四个核心列
        sub = sub[['Feature', 'Prior_Beta', 'Posterior_Beta', 'Certainty']].sort_values('Posterior_Beta', ascending=False)
        sub.to_csv(f'Posterior_Beta_{label}.csv', index=False)
        print(f"接口衔接完成：已生成 {f'Posterior_Beta_{label}.csv'}")

    return posterior_df


post_res = run_bayesian_update_final('2026_MCM_Problem_C_Data.csv', res)