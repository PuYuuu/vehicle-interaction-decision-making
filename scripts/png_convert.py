'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-27 23:28:35
LastEditTime: 2024-10-31 01:01:08
FilePath: /vehicle-interaction-decision-making/scripts/png_convert.py
Copyright 2024 puyu, All Rights Reserved.
'''

import os
import matplotlib.pyplot as plt

def to_str(input) -> str:
    str_elements = [str(element) for element in input]
    result = ' '.join(str_elements)
    
    return result

if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    vehicle_png_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)),
                                    'img', 'vehicle')
    png_files = []
    for entry in os.listdir(vehicle_png_path):
        full_path = os.path.join(vehicle_png_path, entry)
        if os.path.isfile(full_path) and entry.lower().endswith('.png'):
            png_files.append(full_path)
    
    for png_file in png_files:
        convert_file = png_file.split('.')[0] + ".mat.txt"
        png = plt.imread(png_file)
        png_shape = png.shape
        with open(convert_file, 'w', encoding='utf-8') as file:
            file.write('Convert from PNG\n')
            file.write(to_str(png_shape) + '\n')
            
            for row in range(png_shape[0]):
                for col in range(png_shape[1]):
                    file.write(to_str(png[row][col]) + '\n')

