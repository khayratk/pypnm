from heapq import heappush, heappop, heapify


class PriorityQueueUpdateable(object):
    """
    Updateable Priority Queue
    Usage:
    pq = PriorityQueueUpdateable()
    pq.push("hello", 3.0)
    pq.push("bye", 2.0)
    pq.push("bye", 1.0)
    pq.pop()
    pq.pop()

    """
    def __init__(self):
        self._heap = []
        self._dict = dict()

    def _clear_heap(self):
        """
        Removes obsolete entries from heap
        """
        value, key = self._heap[0]
        while (key not in self._dict) or (self._dict[key] != value):
            heappop(self._heap)
            if not self._heap:
                break
            value, key = self._heap[0]

    def pop(self):
        if not self:
            raise IndexError("Queue is empty")

        self._clear_heap()

        value, key = heappop(self._heap)
        del self._dict[key]

        return key, value

    def peek(self):
        if not self:
            raise IndexError("Queue is empty")
        self._clear_heap()
        value, key = self._heap[0]
        return key, value

    def push(self, key, value):
        self._dict[key] = value
        heappush(self._heap, (value, key))

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        self.push(key, value)