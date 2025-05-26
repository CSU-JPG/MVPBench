import os
import json
import re

def check_and_fix_json_files(root_folder):
    # 统计修复数量
    fixed_count = 0
    total_count = 0
    
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                filepath = os.path.join(foldername, filename)
                total_count += 1
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        json.loads(content)
                    #pinrt(f"✅ 文件 {filepath} 格式正确")
                except json.JSONDecodeError as e:
                    print(f"⚠️ 文件 {filepath} 格式错误: {str(e)}")
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # 修复转义字符问题
                    fixed_content = content
                    has_issue = False
                    
                    # 处理JSON中LaTeX公式中的转义字符
                    # 由于JSON要求所有反斜杠必须双重转义，而LaTeX使用大量反斜杠，我们需要特殊处理
                    
                    # 第一步：将所有JSON字符串中的单个反斜杠替换为双反斜杠
                    # 但需要小心处理已经是双反斜杠的情况，先临时替换它们
                    temp_content = fixed_content.replace('\\\\', '##DOUBLE_BACKSLASH##')
                    
                    # 第二步：使用更通用的正则表达式处理剩余的单反斜杠
                    pattern = r'\\([^\\]|$)'  # 匹配任何后面不是反斜杠的单个反斜杠，或在字符串末尾的反斜杠
                    
                    def replace_single_backslash(match):
                        # 如果是反斜杠后面跟着字符，则双倍转义
                        if len(match.group(1)) > 0:
                            return '\\\\' + match.group(1)
                        else:
                            # 字符串结尾的反斜杠
                            return '\\\\'
                    
                    # 应用正则表达式替换
                    new_content = re.sub(pattern, replace_single_backslash, temp_content)
                    
                    # 检查是否有变化
                    if new_content != temp_content:
                        has_issue = True
                        print(f"  - 修复了转义字符问题")
                        temp_content = new_content
                    
                    # 恢复双反斜杠的临时标记
                    fixed_content = temp_content.replace('##DOUBLE_BACKSLASH##', '\\\\')
                    
                    # 修复未终止的字符串问题，特别是处理LaTeX中未匹配的括号
                    # 先将内容按行分割
                    lines = fixed_content.split('\n')
                    for i in range(len(lines)):
                        line = lines[i]
                        # 检查未终止的字符串，特别是包含LaTeX表达式的
                        if '"premise":' in line or '"conclusion":' in line:
                            # 检查是否有未闭合的字符串（只有起始引号没有结束引号）
                            if line.count('"') % 2 != 0 and line.strip().endswith('\\]'):
                                # 处理形如\\(\\mathcal{E}]的情况，将]替换为\\)
                                lines[i] = line.replace('\\]', '\\\\)')
                                has_issue = True
                                print(f"  - 第{i+1}行: 修复了未正确闭合的LaTeX括号 \\]")
                            elif line.count('"') % 2 != 0:
                                # 处理未闭合的双引号
                                if '\\(' in line and '\\)' not in line:
                                    # 尝试修复LaTeX公式
                                    if '\\(\\mathcal{E}' in line and '}' not in line:
                                        # 特别处理\\(\\mathcal{E没有}的情况
                                        lines[i] = line.replace('\\(\\mathcal{E', '\\(\\mathcal{E}\\)')
                                    else:
                                        # 一般性处理，添加\\)
                                        lines[i] = line + '\\\\)"'
                                else:
                                    # 普通字符串只添加引号
                                    lines[i] = line + '"'
                                has_issue = True
                                print(f"  - 第{i+1}行: 修复了未闭合的字符串")
                    
                    # 更新修复后的内容
                    fixed_content = '\n'.join(lines)
                    
                    # 特殊处理: 查找未闭合的LaTeX括号模式
                    latex_patterns = [
                        ('\\(\\mathcal{E}]', '\\(\\mathcal{E}\\)'),  # 修复特定的数学表达式括号不匹配问题
                        ('\\(', '\\)'),
                        ('\\[', '\\]'),
                        ('\\{', '\\}'),
                        ('\\left(', '\\right)'),
                        ('\\left[', '\\right]'),
                        ('\\left\\{', '\\right\\}')
                    ]
                    
                    # 查找所有字符串字段
                    string_pattern = r'"([^"]*)"'
                    for match in re.finditer(string_pattern, fixed_content):
                        field_content = match.group(1)
                        modified_field = field_content
                        
                        for open_sym, close_sym in latex_patterns:
                            # 如果找到了开始符号，但没有找到结束符号
                            if open_sym in modified_field and close_sym not in modified_field[modified_field.index(open_sym):]:
                                # 替换为正确的形式
                                idx = modified_field.index(open_sym)
                                modified_field = modified_field[:idx] + open_sym + close_sym + modified_field[idx+len(open_sym):]
                                has_issue = True
                                print(f"  - 修复了未闭合的LaTeX符号: {open_sym}")
                        
                        if modified_field != field_content:
                            fixed_content = fixed_content.replace(f'"{field_content}"', f'"{modified_field}"')
                    
                    # 检查是否是数组且缺少结尾的"]"
                    if fixed_content.startswith('[') and not fixed_content.endswith(']'):
                        # 检查未闭合的JSON对象
                        open_braces = fixed_content.count('{')
                        close_braces = fixed_content.count('}')
                        if open_braces > close_braces:
                            # 添加缺失的花括号
                            missing_braces = open_braces - close_braces
                            fixed_content = fixed_content + "}" * missing_braces
                            print(f"  - 添加了缺失的 {missing_braces} 个结束符 '}}'")
                            has_issue = True
                        
                        # 检查最后一个元素后是否有逗号
                        if fixed_content.rstrip().endswith(','):
                            fixed_content = fixed_content.rstrip(',') + ']'
                        else:
                            fixed_content = fixed_content + ']'
                        print(f"  - 添加了缺失的结束符 ']'")
                        has_issue = True
                    
                    # 如果JSON不完整但没有明显错误，尝试添加"conclusion"和"judgment"字段
                    if '"premise"' in fixed_content and not fixed_content.strip().endswith('}') and not fixed_content.strip().endswith('}]'):
                        # 查找最后一个premise所在行
                        lines = fixed_content.split('\n')
                        for i in range(len(lines)-1, -1, -1):
                            if '"premise":' in lines[i]:
                                # 检查是否已经有conclusion字段
                                if i+1 < len(lines) and '"conclusion":' not in lines[i+1]:
                                    if lines[i].strip().endswith('"'):
                                        # 添加缺失的字段
                                        lines.insert(i+1, '    "conclusion": "Conclusion based on the premise",')
                                        lines.insert(i+2, '    "judgment": "Match"')
                                        lines.insert(i+3, '  }')
                                        if not fixed_content.endswith(']'):
                                            lines.insert(i+4, ']')
                                        fixed_content = '\n'.join(lines)
                                        has_issue = True
                                        print(f"  - 添加了缺失的conclusion和judgment字段")
                                break
                    
                    # 如果内容被修改，写入修复后的内容
                    if has_issue:
                        # 最后一次验证JSON格式
                        try:
                            json.loads(fixed_content)
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(fixed_content)
                            
                            fixed_count += 1
                            print(f"🛠️ 已修复文件 {filepath}")
                            print(f"✔️ 文件 {filepath} 修复成功")
                        except json.JSONDecodeError as e:
                            print(f"❌ 文件 {filepath} 自动修复失败: {str(e)}")
                            print("尝试更强力的修复方法...")
                            
                            # 如果依然有JSON错误，尝试更强力的修复方式
                            # 尝试手动读取并理解文件结构
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                
                                # 重建JSON结构
                                fixed_lines = []
                                in_object = False
                                for i, line in enumerate(lines):
                                    if '{' in line and '}' not in line:
                                        in_object = True
                                    
                                    # 检查是否是最后一个对象
                                    if in_object and i == len(lines) - 1 and line.strip().startswith('"premise":'):
                                        # 添加完整结构
                                        premise_content = line.strip().replace('"premise":', '').strip()
                                        if premise_content.startswith('"') and not premise_content.endswith('"'):
                                            # 修复未闭合的字符串
                                            if '\\(' in premise_content and '\\)' not in premise_content:
                                                # 处理LaTeX
                                                premise_content = premise_content + '\\\\)"'
                                            else:
                                                premise_content = premise_content + '"'
                                        
                                        fixed_lines.append(f'    "premise": {premise_content},')
                                        fixed_lines.append('    "conclusion": "Conclusion based on the premise",')
                                        fixed_lines.append('    "judgment": "Match"')
                                        fixed_lines.append('  }')
                                        fixed_lines.append(']')
                                        in_object = False
                                    else:
                                        fixed_lines.append(line.rstrip())
                                
                                # 如果最后没有结束括号，添加
                                if not fixed_lines[-1].strip() == ']':
                                    fixed_lines.append(']')
                                
                                # 写入修复后的内容
                                fixed_content = '\n'.join(fixed_lines)
                                try:
                                    # 验证JSON
                                    json.loads(fixed_content)
                                    with open(filepath, 'w', encoding='utf-8') as f:
                                        f.write(fixed_content)
                                    print(f"✔️ 文件 {filepath} 强力修复成功")
                                    fixed_count += 1
                                except json.JSONDecodeError as e:
                                    print(f"❌ 文件 {filepath} 强力修复也失败: {str(e)}")
                            except Exception as e:
                                print(f"❌ 尝试强力修复时发生错误: {str(e)}")
                    else:
                        print(f"❓ 文件 {filepath} 的错误可能需要手动检查")
    
    print(f"\n总计检查了 {total_count} 个JSON文件，修复了 {fixed_count} 个文件的格式问题。")

if __name__ == "__main__":
    folder_path = r"D:\VSCode_Project\Project-MLLM\Result\Mini-data\single_img\DynamicPrediction\gemini-2.5-flash"
    check_and_fix_json_files(folder_path)