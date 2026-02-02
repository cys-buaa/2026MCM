import pandas as pd
import numpy as np

def analyze_voting_consistency(official_csv, model_csv):
    # 1. 加载数据
    df_official = pd.read_csv(official_csv)
    df_model = pd.read_csv(model_csv)
    
    # 清洗官方数据：提取每周淘汰信息
    # 结果列包含 "Eliminated Week X"
    results_list = []
    
    # 遍历赛季和周
    seasons = df_official['season'].unique()
    
    for s in seasons:
        # 确定该赛季的规则
        # 排名法: 1, 2, 28-34; 百分比法: 3-27
        method = 'rank' if (s <= 2 or s >= 28) else 'percent'
        
        # 获取该赛季的所有周（根据模型预测文件确定）
        weeks = df_model[df_model['Season'] == s]['Week'].unique()
        
        for w in weeks:
            # 获取本周模型预测的选手及其数据
            m_week = df_model[(df_model['Season'] == s) & (df_model['Week'] == w)].copy()
            if m_week.empty: continue
            
            # 获取本周真实被淘汰的人
            # 注意：数据集中 results 记录的是该选手最终命运，需匹配当前周
            elim_string = f"Eliminated Week {w}"
            actual_eliminated = df_official[(df_official['season'] == s) & 
                                            (df_official['results'] == elim_string)]['celebrity_name'].tolist()
            
            # 如果这周没有淘汰（如决赛或特殊周），跳过
            if not actual_eliminated: continue
            
            # --- 模拟投票逻辑 ---
            if method == 'rank':
                # 排名法：评委排名 + 粉丝排名 (值越大排名越靠后)
                m_week['j_rank'] = m_week['Judge_Total'].rank(ascending=False, method='min')
                m_week['f_rank'] = m_week['Estimated_Votes'].rank(ascending=False, method='min')
                m_week['final_score'] = m_week['j_rank'] + m_week['f_rank']
                # 预测淘汰：综合排名分最高的选手
                predicted_losers = m_week.nlargest(2, 'final_score')['Celebrity'].tolist()
            else:
                # 百分比法：评委占比 + 粉丝占比
                j_sum = m_week['Judge_Total'].sum()
                f_sum = m_week['Estimated_Votes'].sum()
                m_week['total_pct'] = (m_week['Judge_Total']/j_sum) + (m_week['Fan_Percent'].str.rstrip('%').astype(float)/100)
                # 预测淘汰：百分比最低的选手
                predicted_losers = m_week.nsmallest(2, 'total_pct')['Celebrity'].tolist()

            # --- 比对结果 ---
            pred_first_out = predicted_losers[0]
            is_match = pred_first_out in actual_eliminated
            
            # 检查是否属于 Judge's Save 机制 (第28季后)
            # 如果实际淘汰的是预测的“倒数第二名”，则可能是评委保住了最后一名
            is_judge_save = False
            if not is_match and s >= 28 and len(predicted_losers) > 1:
                if predicted_losers[1] in actual_eliminated:
                    is_judge_save = True

            results_list.append({
                'Season': s,
                'Week': w,
                'Method': method,
                'Actual_Eliminated': actual_eliminated[0],
                'Predicted_Worst': pred_first_out,
                'Is_Match': is_match,
                'Is_Judge_Save_Scenario': is_judge_save
            })

# --- 汇总结果 ---
    summary_df = pd.DataFrame(results_list)
    
    # 计算总体一致性
    total_accuracy = summary_df['Is_Match'].mean()
    
    # 分别计算两种规则的一致性
    # 按照 Method 列进行分组并计算 Is_Match 的平均值
    method_accuracies = summary_df.groupby('Method')['Is_Match'].mean()
    
    # 提取具体数值（增加错误检查，防止某类数据缺失）
    rank_acc = method_accuracies.get('rank', 0.0)
    pct_acc = method_accuracies.get('percent', 0.0)
    
    return summary_df, total_accuracy, rank_acc, pct_acc

# 运行程序并接收新参数
result_table, total_accuracy, rank_accuracy, percent_accuracy = analyze_voting_consistency(
    '2026_MCM_Problem_C_Data.csv', 
    'Weekly_Votes_Based_On_Your_Model.csv'
)

print(f"--- 预测一致性报告 ---")
print(f"百分比法规则一致性 (S3-S27): {percent_accuracy:.2%}")
print(f"排名法规则一致性 (S1-S2, S28+): {rank_accuracy:.2%}")
print(f"----------------------")
print(f"模型预测总一致性: {total_accuracy:.2%}")
# 运行程序
#result_table, total_accuracy = analyze_voting_consistency('2026_MCM_Problem_C_Data.csv', 'Weekly_Votes_Based_On_Your_Model.csv')
#print(f"模型预测总一致性: {total_accuracy:.2%}")