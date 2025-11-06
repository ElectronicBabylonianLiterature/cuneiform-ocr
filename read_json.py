#!/usr/bin/env python3
"""
Simple script to read JSON file and check the length of the first level list
"""

import json
import os

def read_json_and_get_length(file_path):
    """
    Read JSON file and return the length of the first level list
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"错误：文件 {file_path} 不存在")
            return None
        
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"文件大小: {file_size / (1024*1024):.2f} MB")
        
        print("正在读取JSON文件...")
        
        # Read and parse JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("JSON文件读取成功！")
        
        # Check data type and get length
        if isinstance(data, list):
            length = len(data)
            print(f"第一层列表长度: {length}")
            
            # Count total signs from ocredSignsCoordinates
            total_signs = 0
            valid_entries = 0
            
            print("\n正在统计signs数量...")
            
            for i, item in enumerate(data):
                if isinstance(item, dict) and 'ocredSignsCoordinates' in item:
                    if isinstance(item['ocredSignsCoordinates'], list):
                        signs_count = len(item['ocredSignsCoordinates'])
                        total_signs += signs_count
                        valid_entries += 1
                        
                        # Show progress for large datasets
                        if (i + 1) % 10000 == 0:
                            print(f"  已处理 {i + 1}/{length} 个条目...")
                
            print(f"\n统计结果:")
            print(f"有效条目数: {valid_entries}")
            print(f"总signs数量: {total_signs}")
            if valid_entries > 0:
                print(f"平均每个条目的signs数: {total_signs / valid_entries:.2f}")
            
            # Show some basic info about the first few items if they exist
            if length > 0:
                print(f"\n前几个元素的详细信息:")
                for i in range(min(3, length)):
                    print(f"  第{i+1}个元素:")
                    print(f"    类型: {type(data[i])}")
                    if isinstance(data[i], dict):
                        print(f"    字典键: {list(data[i].keys())}")
                        if 'ocredSignsCoordinates' in data[i]:
                            coords_len = len(data[i]['ocredSignsCoordinates']) if isinstance(data[i]['ocredSignsCoordinates'], list) else 'N/A'
                            print(f"    ocredSignsCoordinates长度: {coords_len}")
                        if 'filename' in data[i]:
                            print(f"    filename: {data[i]['filename']}")
                    elif isinstance(data[i], (str, int, float)):
                        print(f"    值: {str(data[i])[:100]}")  # Show first 100 chars
        
        elif isinstance(data, dict):
            print("数据是字典类型，不是列表")
            print(f"字典键数量: {len(data)}")
            print(f"字典的键: {list(data.keys())[:10]}")  # Show first 10 keys
        
        else:
            print(f"数据类型: {type(data)}")
            if hasattr(data, '__len__'):
                print(f"长度: {len(data)}")
        
        return data
    
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None
    except MemoryError:
        print("内存不足，文件太大无法一次性读取")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

if __name__ == "__main__":
    json_file_path = "/home/jebediahc/erc/eBL_OCRed_Signs.json"
    
    print("=" * 50)
    print("JSON文件读取工具")
    print("=" * 50)
    
    data = read_json_and_get_length(json_file_path)
    
    print("=" * 50)
    print("完成")
