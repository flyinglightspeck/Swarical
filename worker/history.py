import time
import heapq
from functools import total_ordering
from config import Config


class History:
    def __init__(self, size):
        self.lists = dict()
        for i in range(size):
            self.lists[i] = []

    def __getitem__(self, item):
        return self.lists[item]

    def log(self, category, value, meta={}):
        if category == 9 or Config.DURATION < 660:
            entry = HistoryEntry(value, meta)
            self.lists[category].append(entry)
            return entry

    def log_sum(self, category):
        self.lists[category][0] += 1

    def merge_lists(self):
        lists = list(self.lists.values())
        heap = []
        for i, lst in enumerate(lists):
            if lst:
                heap.append((lst[0], i, 0))
        heapq.heapify(heap)

        merged = []
        while heap:
            val, lst_idx, elem_idx = heapq.heappop(heap)
            merged.append(lists[lst_idx][elem_idx])
            if elem_idx + 1 < len(lists[lst_idx]):
                next_elem = lists[lst_idx][elem_idx + 1]
                heapq.heappush(heap, (next_elem, lst_idx, elem_idx + 1))
        return merged

    def slice(self, start, end):
        filtered_lists = dict()
        for i in range(len(self.lists)):
            filtered_lists[i] = filter(lambda x: start <= x.t <= end, self.lists[i])


@total_ordering
class HistoryEntry:
    def __init__(self, value, meta):
        self.t = time.time()
        self.meta = meta
        self.value = value

    def __eq__(self, other):
        return self.t == other.t

    def __lt__(self, other):
        return self.t < other.t

    def __repr__(self):
        return f"{self.t} {self.value}"
