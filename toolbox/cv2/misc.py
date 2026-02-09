#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List, Union


def erode(labels: List[Union[str, int]], erode_label: Union[str, int], n: int = 1):
    """
    遍历 labels 列表, 将连续的 erode_label 标签侵蚀 n 个.
    """
    result = list()
    in_span = False
    count = 0
    for idx, label in enumerate(labels):
        if label == erode_label:
            if not in_span:
                in_span = True
                count = 0
            if count < n:
                if len(result) == 0:
                    result.append(label)
                else:
                    result.append(result[-1])
                count += 1
                continue
            else:
                result.append(label)
                continue
        elif label != erode_label:
            if in_span:
                in_span = False

                for i in range(min(len(result), n)):
                    result[-i-1] = label
                result.append(label)
                continue
            else:
                result.append(label)
                continue

        result.append(label)
    return result


def dilate(labels: List[Union[str, int]], dilate_label: Union[str, int], n: int = 1):
    """
    遍历 labels 列表, 将连续的 dilate_label 标签扩张 n 个.
    """
    result = list()
    in_span = False
    count = float('inf')
    for idx, label in enumerate(labels):
        if count < n:
            result.append(dilate_label)
            count += 1
            continue
        if label == dilate_label:
            if not in_span:
                in_span = True

                for i in range(min(len(result), n)):
                    result[-i-1] = label
                result.append(label)
                continue
            else:
                result.append(label)
                continue
        else:
            if in_span:
                in_span = False
                result.append(dilate_label)
                count = 1
                continue
            else:
                result.append(label)
                continue

    return result


def demo1():
    labels = [
        'voice', 'mute', 'mute', 'voice', 'voice', 'voice', 'voice', 'bell', 'bell', 'bell', 'mute', 'mute', 'mute', 'voice',
    ]

    result = erode(
        labels=labels,
        erode_label='voice',
        n=1,

    )
    print(len(labels))
    print(len(result))
    print(result)
    return


def demo2():
    labels = [
        'voice', 'mute', 'mute', 'voice', 'voice', 'voice', 'voice', 'bell', 'bell', 'bell', 'mute', 'mute', 'mute', 'voice',
    ]

    result = dilate(
        labels=labels,
        dilate_label='voice',
        n=2,

    )
    print(len(labels))
    print(len(result))
    print(result)

    return


def demo3():
    import time
    labels = ['mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'voice', 'bell', 'bell', 'bell', 'bell', 'bell', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'bell', 'bell', 'bell', 'bell', 'bell', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'bell', 'bell', 'bell', 'bell', 'bell', 'bell', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute', 'mute']

    begin = time.time()
    labels = erode(labels, erode_label='music', n=1)
    labels = dilate(labels, dilate_label='music', n=1)

    labels = dilate(labels, dilate_label='voice', n=2)
    labels = erode(labels, erode_label='voice', n=2)
    labels = erode(labels, erode_label='voice', n=1)
    labels = dilate(labels, dilate_label='voice', n=3)

    cost = time.time() - begin
    print(cost)
    print(labels)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
