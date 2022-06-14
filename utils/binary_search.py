# -*- coding:utf-8 -*-
# @Organization: Alibaba
# @Author: Yukun Zheng
# @Email: zyk265182@alibaba-inc.com
# @Time: 2020/9/18 14:24

# 返回 x 在 arr 中的索引，如果不存在返回 -1
def binarySearch(arr, l, r, x, func):
    if r >= l:
        mid = int(l + (r - l) / 2)
        # 找到正确位置
        if func(arr, mid) >= x and (mid == 0 or func(arr, mid-1) < x):
            return arr[:mid]
        # 元素小于中间位置的元素，只需要再比较左边的元素
        elif func(arr, mid) >= x:
            return binarySearch(arr, l, mid - 1, x, func)
        # 元素大于中间位置的元素，且右边没有任何元素
        elif func(arr, mid) < x and mid == len(arr) - 1:
            return arr
        # 元素大于中间位置的元素，只需要再比较右边的元素
        else:
            return binarySearch(arr, mid + 1, r, x, func)
    else:
        return []

def get_value(arr, i):
    return arr[i]

if __name__ == '__main__':
    # 测试数组
    arr = [2, 3, 3, 4, 10, 40]
    x = 20
    result = binarySearch(sorted(arr), 0, len(arr) - 1, x, get_value)
    print result
