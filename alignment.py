import numpy as np
import math
def js(A, B):
    cummulative_score = 0
    longest_list = B
    shortest_list = A
    if len(A) > len(B):
        longest_list = A
        shortest_list = B

    for index, element in enumerate(longest_list):
        if index < len(shortest_list):
            if element == shortest_list[index]:
                cummulative_score += 1
            if element != shortest_list[index]:
                if element in shortest_list:
                    cummulative_score += 0.5
    return cummulative_score / len(longest_list)


pred = ['ABZ411 ABZ214 ABZ461 ABZ334 ABZ397 ABZ99 ABZ13 ABZ151 ABZ342 ABZ532 ABZ461 ABZ231 ABZ298 ABZ579 ABZ13 ABZ13 \
        ABZ461 ABZ139 ABZ579', 'ABZ313 ABZ142 ABZ537 ABZ139', 'ABZ142 ABZ579 ABZ401 ABZ6 ABZ167 ABZ328 ABZ70 ABZ579 ABZ449 \
        ABZ75 ABZ68 ABZ335 ABZ73', 'ABZ142 ABZ461 ABZ536 ABZ537 ABZ148 ABZ206 ABZ457 ABZ206 ABZ313 ABZ457 ABZ579 ABZ172 \
        ABZ61 ABZ142 ABZ97 ABZ376 ABZ537 ABZ579 ABZ308 ABZ232 ABZ313 ABZ579 ABZ142 ABZ328', 'ABZ13 ABZ61 ABZ533 ABZ455', 
        'ABZ142 ABZ75 ABZ58 ABZ70 ABZ533 ABZ61 ABZ142 ABZ97 ABZ7 ABZ334 ABZ411', 'ABZ206 ABZ142', 
        'ABZ13 ABZ75 ABZ13 ABZ295 ABZ61 ABZ12 ABZ1 ABZ570 ABZ68', 'ABZ206 ABZ68 ABZ151 ABZ68 ABZ579 ABZ579', 
        'ABZ205 ABZ5 ABZ58 ABZ545 ABZ480 ABZ342 ABZ579 ABZ461', 'ABZ58 ABZ2 ABZ342']
src = ['ABZ554 ABZ536 ABZ57 LU₂@s ABZ532 ABZ232 ABZ13 ABZ461 ABZ579', 'ABZ411 DIM×ŠE ABZ579 ABZ13 ABZ342 ABZ298 ABZ545', 
       'ABZ350 ABZ57 ABZ545 ABZ13 HI×GAD ABZ461 ABZ579 ABZ13 ABZ139', 'ABZ142 ABZ537 ABZ313 ABZ142 ABZ537 ABZ313', 
       'ABZ170 ABZ401 ABZ328 ABZ75 ABZ312 ABZ73', 'ABZ579 ABZ579 ABZ6 ABZ168 ABZ70 ABZ579 ABZ449 ABZ449 ABZ68 ABZ335 ABZ6', 
       'ABZ554 ABZ536 ABZ206 ABZ206 ABZ579 ABZ61 ABZ376 ABZ308 ABZ579', 
       'ABZ554 ABZ536 ABZ148 ABZ457 ABZ457 ABZ172 ABZ43 ABZ579 ABZ367 ABZ232 ABZ328', 
       'ABZ142 ABZ537 ABZ313 ABZ142 ABZ537 ABZ313', 'ABZ13 ABZ70 ABZ61 ABZ312 ABZ352', 'ABZ43 ABZ350 ABZ6 ABZ461', 
       'ABZ148 ABZ457 ABZ58 ABZ328', 'ABZ554 ABZ536 ABZ57 ABZ13 ABZ579 ABZ68 ABZ68', 'ABZ554 ABZ536 ABZ57 ABZ350 ABZ461 ABZ579']
line_alignment_result = np.zeros_like(pred)
for idx, pred_line in enumerate(pred):
    print(pred_line)
    lines_scores = np.zeros(len(src))
    for src_idx, src_line in enumerate(src):
        lines_scores[src_idx] = js(pred_line.split(' '), src_line.split(' '))
    order = lines_scores.argsort(kind='stable')[::-1]
    min_distance = float('inf')
    curr_distance = math.abs(order[0] - idx)
    while curr_distance < min_distance:
        pass
        
   # for i in range(len(ranks)):
    #    if ranks 
    print(lines_scores)
