import json
import os
from pathlib import Path

def calculate_accuracy(jsonl_file):
    """计算JSONL文件中correct=True的比例"""
    total = 0
    correct = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                total += 1
                # 检查correct字段，支持布尔值True/False或字符串"True"/"False"
                correct_value = data.get('correct', data.get('is_correct', False))
                if correct_value in [True, 'True', 'true', 'TRUE']:
                    correct += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: 解析行时出错 {total + 1}: {e}")
                continue
    
    if total == 0:
        return 0.0, 0, 0
    
    accuracy = (correct / total) * 100
    return accuracy, correct, total

def get_display_name(filename):
    """获取显示名称：去掉gsm8k_predictions_前缀和.jsonl后缀"""
    name = filename.replace('.jsonl', '')
    if name.startswith('gsm8k_predictions_'):
        name = name[len('gsm8k_predictions_'):]
    return name

def main():
    # 获取当前目录下所有jsonl文件
    current_dir = Path('.')
    jsonl_files = list(current_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print("当前目录下未找到JSONL文件")
        return
    
    # 过滤只包含gsm8k_predictions的文件（可选）
    gsm8k_files = [f for f in jsonl_files if 'gsm8k_predictions' in f.name]
    
    if gsm8k_files:
        files_to_process = gsm8k_files
        print(f"找到 {len(gsm8k_files)} 个GSM8k预测文件")
    else:
        files_to_process = jsonl_files
        print(f"找到 {len(jsonl_files)} 个JSONL文件")
    
    # 收集结果
    results = []
    for jsonl_file in files_to_process:
        accuracy, correct, total = calculate_accuracy(jsonl_file)
        display_name = get_display_name(jsonl_file.name)
        results.append({
            'file': jsonl_file.name,
            'display_name': display_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        })
    
    # 按准确率降序排序
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # 输出MD表格
    print("\n## 📊 GSM8k 模型评估结果\n")
    print("| 模型名称 | 准确率 | 正确数/总数 |")
    print("|----------|--------|-------------|")
    
    for result in results:
        print(f"| {result['display_name']} | {result['accuracy']:.2f}% | {result['correct']}/{result['total']} |")
    
    # 输出详细统计信息
    print("\n### 📈 详细统计\n")
    print("```")
    for result in results:
        print(f"{result['display_name']}:")
        print(f"  文件: {result['file']}")
        print(f"  准确率: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
        print()
    print("```")
    
    # 保存结果到文件
    with open('gsm8k_evaluation_results.md', 'w', encoding='utf-8') as f:
        f.write("# GSM8k 模型评估结果\n\n")
        f.write("| 模型名称 | 准确率 | 正确数/总数 |\n")
        f.write("|----------|--------|-------------|\n")
        
        for result in results:
            f.write(f"| {result['display_name']} | {result['accuracy']:.2f}% | {result['correct']}/{result['total']} |\n")
    
    print(f"\n结果已保存到: gsm8k_evaluation_results.md")

if __name__ == "__main__":
    main()